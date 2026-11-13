import torch
import numpy as np
from .fm_solvers import (FlowDPMSolverMultistepScheduler)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .basic_flowmatch import FlowMatchScheduler
from .flowmatch_pusa import FlowMatchSchedulerPusa
from .flowmatch_res_multistep import FlowMatchSchedulerResMultistep
from .ersde_scheduler import ERSDEScheduler
from .scheduling_flow_match_lcm import FlowMatchLCMScheduler
from .fm_sa_ode import FlowMatchSAODEStableScheduler
from .fm_rcm import rCMFlowMatchScheduler
from .vitb_unipc import ViBTScheduler
from ...utils import log

try:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
except ImportError:
    FlowMatchEulerDiscreteScheduler = None
    DEISMultistepScheduler = None

scheduler_list = [
    "unipc", "unipc/beta",
    "dpm++", "dpm++/beta",
    "dpm++_sde", "dpm++_sde/beta",
    "euler", "euler/beta",
    "longcat_distill_euler",
    "deis",
    "lcm", "lcm/beta",
    "res_multistep",
    "er_sde",
    "flowmatch_causvid",
    "flowmatch_distill",
    "flowmatch_pusa",
    "multitalk",
    "sa_ode_stable",
    "rcm",
    "vibt_unipc",
]

def _apply_custom_sigmas(sample_scheduler, sigmas, device):
    sample_scheduler.sigmas = sigmas.to(device)
    sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
    sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)

def get_scheduler(scheduler, steps, start_step, end_step, shift, device, transformer_dim=5120, denoise_strength=1.0, sigmas=None, log_timesteps=False, enhance_hf=False, **kwargs):
    timesteps = None
    if sigmas is not None:
        steps = len(sigmas) - 1
    if scheduler == 'vibt_unipc':
        sample_scheduler = ViBTScheduler()
        sample_scheduler.set_parameters(shift=shift)
        sample_scheduler.set_timesteps(steps, device=device)
    elif 'unipc' in scheduler:
        sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif scheduler in ['euler/beta', 'euler', 'longcat_distill_euler']:
        if 'longcat' in scheduler:
            num_distill_sample_steps = 50
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, time_shift_type="linear")
            distill_indices = torch.arange(1, num_distill_sample_steps + 1, dtype=torch.float32)
            distill_indices = (distill_indices * (1000 // num_distill_sample_steps)).round().long()

            inference_indices = torch.linspace(0, num_distill_sample_steps, steps+1)[:-1]
            inference_indices = torch.floor(inference_indices).to(torch.int64)

            sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / 1000
            sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas)
        else:
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
            if sigmas is None:
                sample_scheduler.set_timesteps(steps, device=device)
            else:
                _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif 'dpm' in scheduler:
        if 'sde' in scheduler:
            algorithm_type = "sde-dpmsolver++"
        else:
            algorithm_type = "dpmsolver++"
        sample_scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type=algorithm_type)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, use_beta_sigmas=('beta' in scheduler))
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif scheduler == 'deis':
        sample_scheduler = DEISMultistepScheduler(use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device)
            sample_scheduler.sigmas[-1] = 1e-6
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif 'lcm' in scheduler:
        sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device)
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif 'flowmatch_causvid' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        if transformer_dim == 5120:
            denoising_list = [999, 934, 862, 756, 603, 410, 250, 140, 74]
        else:
            if steps != 4:
                raise ValueError("CausVid 1.3B schedule is only for 4 steps")
            denoising_list = [1000, 750, 500, 250]
        sample_scheduler = FlowMatchScheduler(num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True)
        sample_scheduler.timesteps = torch.tensor(denoising_list)[:steps].to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_distill' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        sample_scheduler = FlowMatchScheduler(
            shift=shift, sigma_min=0.0, extra_one_step=True
        )
        sample_scheduler.set_timesteps(1000, training=True)

        denoising_step_list = torch.tensor([999, 750, 500, 250] , dtype=torch.long)
        temp_timesteps = torch.cat((sample_scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        denoising_step_list = temp_timesteps[1000 - denoising_step_list]
        #print("denoising_step_list: ", denoising_step_list)

        if steps != 4:
            raise ValueError("This scheduler is only for 4 steps")

        sample_scheduler.timesteps = denoising_step_list[:steps].clone().detach().to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_pusa' in scheduler:
        sample_scheduler = FlowMatchSchedulerPusa(shift=shift, sigma_min=0.0, extra_one_step=True)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps+1, denoising_strength=denoise_strength, shift=shift)
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif scheduler == 'res_multistep':
        sample_scheduler = FlowMatchSchedulerResMultistep(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength)
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif scheduler == 'er_sde':
        sample_scheduler = ERSDEScheduler(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength)
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif "sa_ode_stable" in scheduler:
        sample_scheduler = FlowMatchSAODEStableScheduler(shift=shift, **kwargs)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device)
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)
    elif 'rcm' in scheduler:
        sample_scheduler = rCMFlowMatchScheduler()
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, sigma_max=120)
        else:
            _apply_custom_sigmas(sample_scheduler, sigmas, device)

    if timesteps is None:
        timesteps = sample_scheduler.timesteps

    if enhance_hf:
        num_tail_uniform_steps = max(3, min(15, int(len(timesteps) * 0.2))) # Use 20% of steps for uniform tail (minimum 3, maximum 15)
        tail_uniform_start = float(timesteps.max()) * 0.5 # Split at 50% of the timestep range
        tail_uniform_end = 0

        timesteps_uniform_tail = list(np.linspace(tail_uniform_start, tail_uniform_end, num_tail_uniform_steps, dtype=np.float32, endpoint=(tail_uniform_end != 0)))
        timesteps_uniform_tail = [torch.tensor(t, device=device).unsqueeze(0) for t in timesteps_uniform_tail]
        filtered_timesteps = [timestep.unsqueeze(0).to(device)  for timestep in timesteps if timestep > tail_uniform_start]
        timesteps = torch.cat(filtered_timesteps + timesteps_uniform_tail)
        sample_scheduler.timesteps = timesteps
        sample_scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=timesteps.device)])

    steps = len(timesteps)
    # Reject ranges that produce an empty timestep slice — otherwise the
    # sampler's main loop never executes and downstream code (e.g. callbacks
    # referenced after the loop) crashes with a confusing UnboundLocalError.
    if isinstance(start_step, int) and start_step >= steps:
        raise ValueError(
            f"start_step ({start_step}) must be < steps ({steps}); "
            f"the requested range would produce an empty timestep slice"
        )
    if isinstance(end_step, int) and end_step != -1 and end_step > steps:
        raise ValueError(
            f"end_step ({end_step}) must be <= steps ({steps})"
        )
    if (isinstance(start_step, int) and end_step != -1 and start_step >= end_step) or (not isinstance(start_step, int) and start_step != -1 and end_step >= start_step):
        raise ValueError(
            f"start_step ({start_step}) must be less than end_step ({end_step})"
        )

    # Determine start and end indices for slicing
    start_idx = 0
    end_idx = len(timesteps) - 1

    if log_timesteps:
        log.info("------- Scheduler info -------")
        log.info(f"Total timesteps: {timesteps}")

    if isinstance(start_step, float):
        idxs = (sample_scheduler.sigmas <= start_step).nonzero(as_tuple=True)[0]
        if len(idxs) > 0:
            start_idx = idxs[0].item()
    elif isinstance(start_step, int):
        if start_step > 0:
            start_idx = start_step

    if isinstance(end_step, float):
        idxs = (sample_scheduler.sigmas >= end_step).nonzero(as_tuple=True)[0]
        if len(idxs) > 0:
            end_idx = idxs[-1].item()
    elif isinstance(end_step, int):
        if end_step != -1:
            end_idx = end_step - 1

    # Slice timesteps and sigmas once, based on indices
    all_timesteps = timesteps
    timesteps = timesteps[start_idx:end_idx+1]
    sample_scheduler.full_sigmas = sample_scheduler.sigmas.clone()
    sample_scheduler.sigmas = sample_scheduler.sigmas[start_idx:start_idx+len(timesteps)+1]  # always one longer

    if log_timesteps:
        log.info(f"Using timesteps: {timesteps}")
        log.info(f"Using sigmas: {sample_scheduler.sigmas}")
        log.info("------------------------------")

    if hasattr(sample_scheduler, 'timesteps'):
        sample_scheduler.timesteps = timesteps
    setattr(sample_scheduler, 'all_timesteps', all_timesteps)

    return sample_scheduler, timesteps, start_idx, end_idx
