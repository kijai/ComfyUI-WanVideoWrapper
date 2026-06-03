"""Load ComfyUI-native quantized checkpoints (NVFP4 / FP8 mixed) in WanVideoWrapper.

ComfyUI core (>=0.23) ships native NVFP4 + mixed-precision support via
``comfy.quant_ops`` (``TensorCoreNVFP4Layout`` / ``QUANT_ALGOS``) with the FP4/FP8
GEMM kernels provided by ``comfy_kitchen``. Checkpoints in that format store, per
quantized linear: the packed ``weight`` (uint8 for NVFP4 / float8 for FP8), the
scale tensors (``weight_scale``, ``weight_scale_2``), and a per-layer
``comfy_quant`` JSON tensor describing the format, plus a top-level
``_quantization_metadata`` header.

WanVideoWrapper's own loader only knows GGUF and fp8-scaled, so these files don't
load. This module adds support by reconstructing the weights as comfy
``QuantizedTensor`` objects — identical to what ComfyUI core's
``_lazy_load_from_state_dict`` does — so a plain ``nn.Linear`` whose ``.weight`` is
a ``QuantizedTensor`` dispatches to comfy_kitchen's ``scaled_mm_nvfp4`` / FP8 GEMM
via ``__torch_dispatch__``. No new kernels; it reuses what ComfyUI already provides.

Detection is from the state dict alone (presence of ``*.comfy_quant`` keys), so no
safetensors-header plumbing is needed.
"""

import json
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

try:
    from comfy.quant_ops import QUANT_ALGOS, get_layout_class, QuantizedTensor, _CK_AVAILABLE
    _COMFY_QUANT_AVAILABLE = True
except Exception as e:  # pragma: no cover - comfy always present in a ComfyUI install
    _COMFY_QUANT_AVAILABLE = False
    QuantizedTensor = None
    logging.getLogger(__name__).warning(f"comfy.quant_ops unavailable, NVFP4 support off: {e}")


def is_comfy_quant_state_dict(sd) -> bool:
    """True if ``sd`` is a ComfyUI-native quantized checkpoint (NVFP4/FP8 mixed)."""
    if not _COMFY_QUANT_AVAILABLE:
        return False
    return any(k.endswith(".comfy_quant") for k in sd)


def _decode_comfy_quant(t: torch.Tensor) -> dict:
    return json.loads(bytes(t.to(torch.uint8).cpu().numpy().tobytes()).decode("utf-8"))


def _build_quantized_tensor(sd, prefix, device, compute_dtype, out_features, in_features):
    """Reconstruct one layer's QuantizedTensor exactly like comfy core's loader."""
    fmt = _decode_comfy_quant(sd[prefix + "comfy_quant"])["format"]
    qcfg = QUANT_ALGOS[fmt]
    layout_name = qcfg["comfy_tensor_layout"]
    layout = get_layout_class(layout_name)

    weight = sd[prefix + "weight"].to(device=device, dtype=qcfg["storage_t"])
    # Derive the LOGICAL shape from the packed weight, not from module.in_features:
    # NVFP4 packs two FP4 values per uint8 byte, so the stored weight is
    # (out, in/2). When a model's Linear was instantiated from the checkpoint's
    # packed weight shape, module.in_features is the *packed* width (in/2) for NVFP4
    # — using it for orig_shape makes dequantize yield a half-width tensor and the
    # matmul fails. FP8 stores one value per byte so packed == logical (that path was
    # always fine). Compute orig_shape from qdata + the format's packing factor.
    out_f = weight.shape[0]
    in_f = weight.shape[1] * (2 if fmt == "nvfp4" else 1)
    if fmt == "nvfp4":
        ts = sd[prefix + "weight_scale_2"].to(device=device)
        bs = sd[prefix + "weight_scale"].to(device=device).view(dtype=torch.float8_e4m3fn)
        params = layout.Params(scale=ts, block_scale=bs, orig_dtype=compute_dtype,
                               orig_shape=(out_f, in_f))
    else:  # float8_e4m3fn / float8_e5m2
        sc = sd[prefix + "weight_scale"].to(device=device)
        params = layout.Params(scale=sc, orig_dtype=compute_dtype,
                               orig_shape=(out_f, in_f))
    return QuantizedTensor(weight, layout_name, params), fmt


def replace_with_comfy_quant_linear(model, sd, compute_dtype, load_device, prefix=""):
    """Walk ``model``; for every ``nn.Linear`` that has a ``comfy_quant`` sibling in
    ``sd``, assign a ``QuantizedTensor`` weight (NVFP4/FP8). Non-quantized linears are
    left for the normal loader. Returns the (mutated) model.

    NVFP4 weights are loaded on ``load_device`` directly (NVFP4's whole point is that
    the model fits); block-swap-aware placement and LoRA-merge are TODO (see PR notes)
    and follow the same "no merge for quantized weights" rule as GGUF / scaled-fp8.
    """
    if not _COMFY_QUANT_AVAILABLE:
        return model

    for name, module in model.named_children():
        child_prefix = (prefix + name + ".").replace("_orig_mod.", "")
        replace_with_comfy_quant_linear(module, sd, compute_dtype, load_device, child_prefix)

        if isinstance(module, nn.Linear) and (child_prefix + "comfy_quant") in sd and "loras" not in child_prefix:
            out_features, in_features = module.out_features, module.in_features
            qt, fmt = _build_quantized_tensor(sd, child_prefix, load_device, compute_dtype, out_features, in_features)
            module.weight = nn.Parameter(qt, requires_grad=False)
            bias_key = child_prefix + "bias"
            if module.bias is not None and bias_key in sd:
                module.bias = nn.Parameter(sd[bias_key].to(device=load_device, dtype=compute_dtype), requires_grad=False)
            module._comfy_quant_format = fmt

            # WanVideoWrapper's CustomLinear routes through a registered custom op
            # (torch.ops.wanvideo.linear_forward) for torch.compile compatibility, but
            # a custom-op boundary strips tensor subclasses -> the QuantizedTensor's
            # __torch_dispatch__ never fires and the raw packed NVFP4 bytes leak into
            # F.linear (wrong shape). Route quantized layers through the DIRECT path,
            # where plain F.linear -> aten.linear.default dispatches to comfy_kitchen's
            # NVFP4/FP8 GEMM. Also clear scale_weight (the scale lives in the
            # QuantizedTensor) and the gguf flag.
            if hasattr(module, "_linear_forward_direct"):
                module._linear_forward_impl = module._linear_forward_direct
                module.scale_weight = None
                module.is_gguf = False

    return model
