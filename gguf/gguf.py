import torch
import numpy as np
import gguf

from .gguf_utils import GGUFParameter

def load_gguf(model_path):
    reader = gguf.GGUFReader(model_path)
    parsed_parameters = {}
    for tensor in reader.tensors:
        # if the tensor is a torch supported dtype do not use GGUFParameter
        is_gguf_quant = tensor.tensor_type not in [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16]
        meta_tensor = torch.empty(tensor.data.shape, dtype=torch.from_numpy(np.empty(0, dtype=tensor.data.dtype)).dtype, device='meta')
        parsed_parameters[tensor.name] = GGUFParameter(meta_tensor, quant_type=tensor.tensor_type) if is_gguf_quant else meta_tensor
    return parsed_parameters, reader

from ..custom_linear import _replace_linear, set_lora_params, CustomLinear

def _replace_with_gguf_linear(model, compute_dtype, state_dict, prefix="", modules_to_not_convert=[], patches=None, compile_args=None):
    return _replace_linear(model, compute_dtype, state_dict, prefix, patches, None, compile_args, modules_to_not_convert)

def set_lora_params_gguf(module, patches, module_prefix="", device=torch.device("cpu"), force_cpu=False, _diag=None):
    return set_lora_params(module, patches, module_prefix, device, force_cpu, _diag=_diag)

GGUFLinear = CustomLinear