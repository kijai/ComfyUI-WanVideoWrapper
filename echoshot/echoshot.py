import torch
from comfy.model_management import get_autocast_device, get_torch_device

def _rope_real_dtype(device):
    return torch.float32 if device.type == "mps" else torch.float64

def _rope_freqs_to(freqs, target):
    dtype = target.dtype if target.is_complex() else None
    return freqs.to(device=target.device, dtype=dtype)

@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply_z(x, grid_sizes, freqs, inner_t, shift=6):
    n, c = x.size(2), x.size(3) // 2

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(_rope_real_dtype(x.device)).reshape(seq_len, n, -1, 2)
        )
        start_ind = [sum(inner_t[i][:_]) for _ in range(len(inner_t[i]))]
        end_ind = [sum(inner_t[i][:_+1]) for _ in range(len(inner_t[i]))]

        freq_select = []
        for shot_ind, (s, e) in enumerate(zip(start_ind, end_ind)):
            freq_select += [shot_ind * shift] * (e - s)
        shot_freqs = freqs[freq_select]

        freqs_i = shot_freqs.view(f, 1, 1, -1).expand(f, h, w, -1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        freqs_i = _rope_freqs_to(freqs_i, x_i)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply_c(x, freqs, inner_c, shift=6):

    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # loop over samples
    output = []
    for i in range(b):

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i].to(_rope_real_dtype(x.device)).reshape(s, n, -1, 2)
        )

        freq_select = []
        for shot_ind, c_len in enumerate(inner_c[i]):
            freq_select += [shot_ind * shift] * c_len
        freq_select += [shot_ind+10] * (s-len(freq_select)) # extra suppression for the empty token
        shot_freqs = freqs[freq_select]

        freqs_i = shot_freqs.view(s, 1, -1)

        # apply rotary embedding
        freqs_i = _rope_freqs_to(freqs_i, x_i)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply_echoshot(x, grid_sizes, freqs, inner_t, shift=4):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(_rope_real_dtype(x.device)).reshape(seq_len, n, -1, 2)
        )
        start_ind = [sum(inner_t[i][:_]) for _ in range(len(inner_t[i]))]
        end_ind = [sum(inner_t[i][:_+1]) for _ in range(len(inner_t[i]))]
        freq_select = []
        for shot_ind, (s, e) in enumerate(zip(start_ind, end_ind)):
            freq_select += list(range(shot_ind * shift + s, shot_ind * shift + e))
        t_freqs = freqs[0][freq_select]

        freqs_i = torch.cat([
            # freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            t_freqs.view(f, 1, 1, -1).expand(f, h, w, -1), ###
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        freqs_i = _rope_freqs_to(freqs_i, x_i)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()
