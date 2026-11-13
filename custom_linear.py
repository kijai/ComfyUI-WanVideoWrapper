import torch, gc
import torch.nn as nn
from accelerate import init_empty_weights
from .gguf.gguf_utils import GGUFParameter, dequantize_gguf_tensor
import logging
_ram_log = logging.getLogger(__name__)

@torch.library.custom_op("wanvideo::apply_lora", mutates_args=())
def apply_lora(weight: torch.Tensor, lora_diff_0: torch.Tensor, lora_diff_1: torch.Tensor, lora_diff_2: float, lora_strength: torch.Tensor) -> torch.Tensor:
    patch_diff = torch.mm(
        lora_diff_0.flatten(start_dim=1),
        lora_diff_1.flatten(start_dim=1)
    ).reshape(weight.shape)

    alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
    scale = lora_strength * alpha

    return weight + patch_diff * scale

@apply_lora.register_fake
def _(weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
    # Return weight with same metadata
    return weight.clone()

@torch.library.custom_op("wanvideo::apply_single_lora", mutates_args=())
def apply_single_lora(weight: torch.Tensor, lora_diff: torch.Tensor, lora_strength: torch.Tensor) -> torch.Tensor:
    return weight + lora_diff * lora_strength

@apply_single_lora.register_fake
def _(weight, lora_diff, lora_strength):
    # Return weight with same metadata
    return weight.clone()

@torch.library.custom_op("wanvideo::linear_forward", mutates_args=())
def linear_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return torch.nn.functional.linear(input, weight, bias)

@linear_forward.register_fake
def _(input, weight, bias):
    # Calculate output shape: (..., out_features)
    out_features = weight.shape[0]
    output_shape = list(input.shape[:-1]) + [out_features]
    return input.new_empty(output_shape)

#based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py
def _replace_linear(model, compute_dtype, state_dict, prefix="", patches=None, scale_weights=None, compile_args=None, modules_to_not_convert=[]):

    has_children = list(model.children())
    if not has_children:
        return

    allow_compile = False

    for name, module in model.named_children():
        if compile_args is not None:
            allow_compile = compile_args.get("allow_unmerged_lora_compile", False)
        module_prefix = prefix + name + "."
        module_prefix = module_prefix.replace("_orig_mod.", "")
        _replace_linear(module, compute_dtype, state_dict, module_prefix, patches, scale_weights, compile_args, modules_to_not_convert)

        if isinstance(module, nn.Linear) and "loras" not in module_prefix and "dual_controller" not in module_prefix and name not in modules_to_not_convert:
            weight_key = module_prefix + "weight"
            if state_dict is not None:
                if weight_key not in state_dict:
                    continue
                weight = state_dict[weight_key]
            else:
                # sd was released to save memory; fall back to the already-loaded parameter
                weight = getattr(module, "weight", None)
                if weight is None or weight.numel() == 0:
                    continue

            in_features = weight.shape[1]
            out_features = weight.shape[0]

            is_gguf = isinstance(weight, GGUFParameter)

            scale_weight = None
            if not is_gguf and scale_weights is not None:
                scale_key = f"{module_prefix}scale_weight"
                scale_weight = scale_weights.get(scale_key)

            with init_empty_weights():
                model._modules[name] = CustomLinear(
                    in_features,
                    out_features,
                    module.bias is not None,
                    compute_dtype=compute_dtype,
                    scale_weight=scale_weight,
                    allow_compile=allow_compile,
                    is_gguf=is_gguf
                )
            model._modules[name].source_cls = type(module)
            model._modules[name].requires_grad_(False)

    return model

def set_lora_params(module, patches, module_prefix="", device=torch.device("cpu"), force_cpu=False, _depth=0, _diag=None):
    """Apply LoRA patches to CustomLinear layers using a progressive approach.

    Instead of recursively iterating modules and holding all float32 LoRA tensors
    in memory throughout, this function:
    1. Clears any previously-applied LoRA attributes
    2. Builds a key→CustomLinear map once via named_modules()
    3. Iterates patches dict, applies each patch immediately, then DELETES
       the entry from patches to free float32 originals progressively

    This avoids the peak memory scenario where ALL float32 originals (~16 GB)
    AND ALL bfloat16 copies (~8 GB) coexist simultaneously.

    Returns (lora_param_count, lora_total_bytes, module_count_matched).
    When _diag is a dict, fills it with diagnostic counters:
      _diag['customlinear_total'] = total CustomLinear modules found
      _diag['customlinear_matched'] = number matched with a patch
      _diag['customlinear_bytes'] = bytes of bfloat16 LoRA tensors stored
      _diag['_key_mismatches'] = first 5 patch keys with no matching module
    """
    import psutil, os as _os
    _pid = _os.getpid()
    def _rss_mb():
        try:
            return psutil.Process(_pid).memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    if _diag is None:
        _diag = {}

    _rss0 = _rss_mb()
    _ram_log.info(f"[RAM-diag] set_lora_params start | RSS: {_rss0:.1f} MB | {len(patches)} patch entries")

    # Step 1: Clear any previously applied LoRA attrs from all CustomLinear modules
    remove_lora_from_module(module)
    _rss1 = _rss_mb()
    _ram_log.info(f"[RAM-diag] after remove_lora_from_module | RSS: {_rss1:.1f} MB (delta: {_rss1 - _rss0:+.1f} MB)")

    # Step 2: Build key→CustomLinear map once
    # named_modules() returns (name, module) where name is the full dotted path
    cl_map = {}
    for mod_name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear):
            # Construct key matching patcher.patches convention
            key = f"diffusion_model.{mod_name}.weight"
            cl_map[key] = submodule
            # Also register the _orig_mod stripped variant (from torch.compile wrapping)
            stripped = key.replace("_orig_mod.", "")
            if stripped != key:
                cl_map[stripped] = submodule

    _unique_cl = len(set(id(m) for _, m in cl_map.items()))
    _rss2 = _rss_mb()
    _ram_log.info(f"[RAM-diag] after building CL map ({len(cl_map)} key entries, {_unique_cl} unique modules) | RSS: {_rss2:.1f} MB")

    # Step 3a: Eject ALL unmatched patches FIRST — frees float32 tensors
    #           before any bfloat16 conversion starts, eliminating the
    #           intermediate RSS peak from ~650 leftover float32 patches.
    unmatched_keys = [k for k in patches if k not in cl_map]
    _diag.setdefault('_key_mismatches', [])
    for k in unmatched_keys[:5]:
        _diag['_key_mismatches'].append(k)
    for k in unmatched_keys:
        del patches[k]
    unmatched_count = len(unmatched_keys)
    if unmatched_count:
        _rss_pre = _rss_mb()
        gc.collect()
        _rss_post = _rss_mb()
        _ram_log.info(
            f"[RAM-diag] ejected {unmatched_count} unmatched patches | "
            f"RSS: {_rss_pre:.1f} → {_rss_post:.1f} MB (delta: {_rss_post - _rss_pre:+.1f} MB)"
        )

    # Step 3b: Process matched patches progressively
    lora_param_count = 0
    lora_total_bytes = 0
    module_count_matched = 0
    total_patches = len(patches)
    processed = 0

    for key in list(patches.keys()):
        cl_module = cl_map[key]
        patch = patches[key]

        # Build LoRA diff list from patch entries.
        # PRE-CONVERT to (CPU, compute_dtype) so set_lora_diffs sees
        # already-matching tensors and returns self (zero-copy).
        # Without explicit device="cpu", float32 CUDA tensors get
        # converted to bf16-CUDA, then set_lora_diffs creates a SECOND
        # copy via .to(cpu, bf16) — doubling RSS.
        target_device = torch.device("cpu")
        lora_diffs = []
        for p in patch:
            lora_obj = p[1]
            if "head" in key:
                continue  # Skip LoRA for head layers
            elif hasattr(lora_obj, "weights"):
                weights = lora_obj.weights
                new_weights = tuple(
                    w.to(device=target_device, dtype=cl_module.compute_dtype)
                    if torch.is_tensor(w) else w
                    for w in weights
                )
                lora_obj.weights = new_weights
                lora_diffs.append(new_weights)
            elif isinstance(lora_obj, tuple) and lora_obj[0] == "diff":
                diffs = lora_obj[1]
                new_diffs = tuple(
                    w.to(device=target_device, dtype=cl_module.compute_dtype)
                    if torch.is_tensor(w) else w
                    for w in diffs
                )
                lora_diffs.append(new_diffs)
            else:
                continue

        if not lora_diffs:
            del patches[key]
            continue

        lora_strengths = [p[0] for p in patch]
        diff_bytes = cl_module.set_lora_diffs(lora_diffs, device=device, _diag=_diag)
        cl_module.set_lora_strengths(lora_strengths, device=device)
        cl_module._step.fill_(0)

        module_count_matched += 1
        lora_total_bytes += diff_bytes
        lora_param_count += len(lora_diffs)

        # IMMEDIATELY delete from patches dict to free float32 originals
        del patches[key]
        processed += 1

        # Periodic GC to help OS reclaim freed pages
        if processed % 50 == 0:
            gc.collect()
            _rss_now = _rss_mb()
            _ram_log.info(f"[RAM-diag] progress {processed}/{total_patches} | RSS: {_rss_now:.1f} MB | accum bytes: {lora_total_bytes / (1024**2):.1f} MB")

    # Final cleanup
    gc.collect()

    _diag['customlinear_total'] = _unique_cl  # unique CustomLinear modules, not key entries
    _diag['customlinear_matched'] = module_count_matched
    _diag['customlinear_bytes'] = lora_total_bytes

    _rss3 = _rss_mb()
    _ram_log.info(
        f"[RAM-diag] set_lora_params done | RSS: {_rss3:.1f} MB (delta: {_rss3 - _rss0:+.1f} MB) | "
        f"total CL: {_unique_cl} | matched: {module_count_matched} | "
        f"accum_bytes: {lora_total_bytes / (1024**2):.1f} MB | "
        f"to_copies={_diag.get('to_copies', '?')} to_noops={_diag.get('to_noops', '?')} | "
        f"d2_bytes={_diag.get('d2_total_bytes', 0) / (1024**2):.1f} MB (count={_diag.get('d2_count', 0)})"
    )

    return lora_param_count, lora_total_bytes, module_count_matched

class CustomLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        compute_dtype=None,
        device=None,
        scale_weight=None,
        allow_compile=False,
        is_gguf=False
    ) -> None:
        super().__init__(in_features, out_features, bias, device)
        self.compute_dtype = compute_dtype
        self.lora_diffs = []
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.scale_weight = scale_weight
        self.lora_strengths = []
        self.allow_compile = allow_compile
        self.is_gguf = is_gguf

        if not allow_compile:
            self._apply_lora_impl = self._apply_lora_custom_op
            self._apply_single_lora_impl = self._apply_single_lora_custom_op
            self._linear_forward_impl = self._linear_forward_custom_op
        else:
            self._apply_lora_impl = self._apply_lora_direct
            self._apply_single_lora_impl = self._apply_single_lora_direct
            self._linear_forward_impl = self._linear_forward_direct


    # Direct implementations (no custom ops)
    def _apply_lora_direct(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        patch_diff = torch.mm(
            lora_diff_0.flatten(start_dim=1),
            lora_diff_1.flatten(start_dim=1)
        ).reshape(weight.shape) + 0
        alpha = lora_diff_2 / lora_diff_1.shape[0] if lora_diff_2 != 0.0 else 1.0
        scale = lora_strength * alpha
        return weight + patch_diff * scale

    def _apply_single_lora_direct(self, weight, lora_diff, lora_strength):
        return weight + lora_diff * lora_strength

    def _linear_forward_direct(self, input, weight, bias):
        return torch.nn.functional.linear(input, weight, bias)

    # Custom op implementations
    def _apply_lora_custom_op(self, weight, lora_diff_0, lora_diff_1, lora_diff_2, lora_strength):
        return torch.ops.wanvideo.apply_lora(weight, lora_diff_0, lora_diff_1,
            float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
        )

    def _apply_single_lora_custom_op(self, weight, lora_diff, lora_strength):
        return torch.ops.wanvideo.apply_single_lora(weight, lora_diff, lora_strength)

    def _linear_forward_custom_op(self, input, weight, bias):
        return torch.ops.wanvideo.linear_forward(input, weight, bias)

    def set_lora_diffs(self, lora_diffs, device=torch.device("cpu"), _diag=None):
        self.lora_diffs = []
        diff_bytes = 0
        if _diag is None:
            _diag = {}
        for i, diff in enumerate(lora_diffs):
            if len(diff) > 1:
                d0_src = diff[0]
                d1_src = diff[1]
                d2_src = diff[2]
                d0 = d0_src.to(device, self.compute_dtype)
                d1 = d1_src.to(device, self.compute_dtype)
                # Diagnostic: check if .to() created a copy
                _diag.setdefault('to_copies', 0)
                _diag.setdefault('to_noops', 0)
                if d0 is d0_src:
                    _diag['to_noops'] += 1
                else:
                    _diag['to_copies'] += 1
                if d1 is d1_src:
                    _diag['to_noops'] += 1
                else:
                    _diag['to_copies'] += 1
                # Check diff[2] size
                if torch.is_tensor(d2_src):
                    d2_bytes = d2_src.numel() * d2_src.element_size()
                    _diag.setdefault('d2_total_bytes', 0)
                    _diag['d2_total_bytes'] += d2_bytes
                    _diag.setdefault('d2_count', 0)
                    _diag['d2_count'] += 1
                setattr(self, f"lora_diff_{i}_0", d0)
                setattr(self, f"lora_diff_{i}_1", d1)
                setattr(self, f"lora_diff_{i}_2", d2_src)
                self.lora_diffs.append((f"lora_diff_{i}_0", f"lora_diff_{i}_1", f"lora_diff_{i}_2"))
                diff_bytes += d0.numel() * d0.element_size() + d1.numel() * d1.element_size()
            else:
                d0_src = diff[0]
                d0 = d0_src.to(device, self.compute_dtype)
                _diag.setdefault('to_copies', 0)
                _diag.setdefault('to_noops', 0)
                if d0 is d0_src:
                    _diag['to_noops'] += 1
                else:
                    _diag['to_copies'] += 1
                setattr(self, f"lora_diff_{i}_0", d0)
                self.lora_diffs.append(f"lora_diff_{i}_0")
                diff_bytes += d0.numel() * d0.element_size()
        return diff_bytes

    def set_lora_strengths(self, lora_strengths, device=torch.device("cpu")):
        self._lora_strength_tensors = []
        self._lora_strength_is_scheduled = []
        self._step = self._step.to(device)
        for i, strength in enumerate(lora_strengths):
            if isinstance(strength, list):
                tensor = torch.tensor(strength, dtype=self.compute_dtype, device=device)
                setattr(self, f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(True)
            else:
                tensor = torch.tensor([strength], dtype=self.compute_dtype, device=device)
                setattr(self, f"_lora_strength_{i}", tensor)
                self._lora_strength_is_scheduled.append(False)

    def _get_lora_strength(self, idx):
        strength_tensor = getattr(self, f"_lora_strength_{idx}")
        if self._lora_strength_is_scheduled[idx]:
            return strength_tensor.index_select(0, self._step).squeeze(0)
        return strength_tensor[0]

    def _get_weight_with_lora(self, weight):
        """Apply LoRA using custom ops to avoid graph breaks"""
        if not hasattr(self, "lora_diff_0_0"):
            return weight

        for idx, lora_diff_names in enumerate(self.lora_diffs):
            lora_strength = self._get_lora_strength(idx)
            if lora_strength.device != weight.device:
                lora_strength = lora_strength.to(weight.device, weight.dtype)

            if isinstance(lora_diff_names, tuple):
                lora_diff_0 = getattr(self, lora_diff_names[0])
                lora_diff_1 = getattr(self, lora_diff_names[1])
                lora_diff_2 = getattr(self, lora_diff_names[2])
                if lora_diff_0.device != weight.device:
                    lora_diff_0 = lora_diff_0.to(weight.device, weight.dtype)
                if lora_diff_1.device != weight.device:
                    lora_diff_1 = lora_diff_1.to(weight.device, weight.dtype)

                weight = self._apply_lora_impl(
                    weight, lora_diff_0, lora_diff_1,
                    float(lora_diff_2) if lora_diff_2 is not None else 0.0, lora_strength
                )
            else:
                lora_diff = getattr(self, lora_diff_names)
                if lora_diff.device != weight.device:
                    lora_diff = lora_diff.to(weight.device, weight.dtype)
                weight = self._apply_single_lora_impl(weight, lora_diff, lora_strength)
        return weight

    def _prepare_weight(self, input):
        """Prepare weight tensor - handles both regular and GGUF weights"""
        if self.is_gguf:
            weight = dequantize_gguf_tensor(self.weight).to(self.compute_dtype)
        else:
            weight = self.weight.to(input)
        return weight

    def forward(self, input):
        weight = self._prepare_weight(input)

        if self.bias is not None:
            bias = self.bias.to(input if not self.is_gguf else self.compute_dtype)
        else:
            bias = None

        # Only apply scale_weight for non-GGUF models
        if not self.is_gguf and self.scale_weight is not None:
            if weight.numel() < input.numel():
                weight = weight * self.scale_weight
            else:
                input = input * self.scale_weight

        weight = self._get_weight_with_lora(weight)
        out = self._linear_forward_impl(input, weight, bias)
        del weight, input, bias
        return out

def update_lora_step(module, step):
    for name, submodule in module.named_modules():
        if isinstance(submodule, CustomLinear) and hasattr(submodule, "_step"):
            submodule._step.fill_(step)

def remove_lora_from_module(module):
    for name, submodule in module.named_modules():
        if hasattr(submodule, "lora_diffs"):
            for i in range(len(submodule.lora_diffs)):
                if hasattr(submodule, f"lora_diff_{i}_0"):
                    delattr(submodule, f"lora_diff_{i}_0")
                if hasattr(submodule, f"lora_diff_{i}_1"):
                    delattr(submodule, f"lora_diff_{i}_1")
                if hasattr(submodule, f"lora_diff_{i}_2"):
                    delattr(submodule, f"lora_diff_{i}_2")
            # Clear strength tensors as well
            i = 0
            while hasattr(submodule, f"_lora_strength_{i}"):
                delattr(submodule, f"_lora_strength_{i}")
                i += 1
            submodule.lora_diffs = []
            submodule._lora_strength_is_scheduled = []
