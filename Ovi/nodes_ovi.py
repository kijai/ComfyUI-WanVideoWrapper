import torch
import torch.nn as nn
import folder_paths
import os

from .mel_converter import get_mel_converter
from .vae.autoencoder import AutoEncoderModule
from .vae.distributions import DiagonalGaussianDistribution
import torchaudio

from ..utils import log

from comfy import model_management as mm
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        tod_vae_ckpt: str,
        bigvgan_vocoder_ckpt = None,
        mode=['16k', '44k'],
        need_vae_encoder: bool = True,
    ):
        super().__init__()

        self.mel_converter = get_mel_converter(mode)
        self.tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                        vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                        mode=mode,
                                        need_vae_encoder=need_vae_encoder)

    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, 'VAE is not loaded'
        # x: (B * L)
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.vocode(mel)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, 'VAE is not loaded'
        return self.tod.decode(z)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def wrapped_decode(self, z):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            mel_decoded = self.decode(z)
            audio = self.vocode(mel_decoded)

            return audio

    def wrapped_encode(self, audio):
        with torch.amp.autocast('cuda', dtype=self.dtype):
            dist = self.encode_audio(audio)

            return dist.mean

if not "mmaudio" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("mmaudio", os.path.join(folder_paths.models_dir, "mmaudio"))

class OviMMAudioVAELoader:
    """Loads MMAudio VAE for audio encoding/decoding in Ovi"""
    @classmethod
    def INPUT_TYPES(s):
        s.vae_files = folder_paths.get_filename_list("vae")
        s.mmaudio_files = folder_paths.get_filename_list("mmaudio")
        s.all_files = s.vae_files + s.mmaudio_files

        return {
            "required": {
                "vae": (s.all_files, {"tooltip": "MMAudio VAE 16k (v1-16.pth) model from models/vae or models/mmaudio"}),
                "vocoder": (s.all_files, {"tooltip": "BigVGAN vocoder (best_netG.pt) from models/vae or models/mmaudio"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": "Compute dtype the MMAudio VAE+vocoder load at; bf16 is the default and matches Ovi's training precision"}),
            }
        }

    RETURN_TYPES = ("MMAUDIOVAE",)
    RETURN_NAMES = ("mmaudio_vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper/Ovi"
    DESCRIPTION = "Loads MMAudio VAE for Ovi audio generation"

    def loadmodel(self, vae, vocoder, precision):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        vae_path = folder_paths.get_full_path("vae", vae) if vae in self.vae_files else folder_paths.get_full_path("mmaudio", vae)
        vocoder_path = folder_paths.get_full_path("vae", vocoder) if vocoder in self.vae_files else folder_paths.get_full_path("mmaudio", vocoder)

        vae = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            bigvgan_vocoder_ckpt=vocoder_path,
            mode='16k',
            need_vae_encoder=True
        )

        vae.to(device=offload_device, dtype=dtype)
        vae.eval()

        return (vae,)

class WanVideoDecodeOviAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mmaudio_vae": ("MMAUDIOVAE", {"tooltip": "MMAudio VAE+vocoder bundle — connect from OviMMAudioVAELoader"}),
                    "samples": ("LATENT", {"tooltip": "Sampled latents dict containing an 'latent_ovi_audio' tensor; decoded back to a 16 kHz waveform via the MMAudio VAE+vocoder"}),
                }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, mmaudio_vae, samples):
        mm.soft_empty_cache()
        audio_latents = samples.get("latent_ovi_audio", None)
        if audio_latents is None:
            raise ValueError("No Ovi audio latents found in input samples")

        mmaudio_vae.to(device)

        waveform = mmaudio_vae.wrapped_decode(audio_latents.to(device=device, dtype=mmaudio_vae.dtype))
        audio = {"waveform": waveform.cpu().float(), "sample_rate": 16000}

        mmaudio_vae.to(offload_device)
        mm.soft_empty_cache()

        return (audio,)

class WanVideoEncodeOviAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mmaudio_vae": ("MMAUDIOVAE", {"tooltip": "MMAudio VAE+vocoder bundle — connect from OviMMAudioVAELoader"}),
                    "audio": ("AUDIO", {"tooltip": "Reference audio waveform; resampled to 16 kHz mono and encoded into the MMAudio latent space for use as audio conditioning"}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, mmaudio_vae, audio):

        mmaudio_vae.to(device)

        waveform = audio.get("waveform", None)
        sample_rate = audio.get("sample_rate", None)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform.to(device=device, dtype=mmaudio_vae.dtype)[0][0].unsqueeze(0)

        samples = mmaudio_vae.wrapped_encode(waveform)

        mmaudio_vae.to(offload_device)
        mm.soft_empty_cache()

        return ({"latent_ovi_audio": samples},)


class WanVideoAddOviAudioToLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "original_samples": ("LATENT", {"tooltip": "Existing latent dict (typically video latents from a sampler); audio_samples keys are merged on top to attach MMAudio latents"}),
                    "audio_samples": ("LATENT", {"tooltip": "MMAudio latent dict (carries 'latent_ovi_audio') — connect from WanVideoEncodeOviAudio or WanVideoEmptyMMAudioLatents"}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, original_samples, audio_samples):
        samples = original_samples.copy()
        samples.update(audio_samples)

        return (samples,)
    
class WanVideoEmptyMMAudioLatents:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "length": ("INT", {"default": 157, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of MMAudio latent timesteps (16 kHz mel frames) to allocate; ~157 = 5 s of audio at Ovi's default rate"}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoWrapper/Ovi"

    def decode(self, length):
        audio_latents = torch.zeros((length, 20), device=torch.device("cpu"), dtype=torch.float32)  # 1, l c -> l, c

        return ({"latent_ovi_audio": audio_latents},)


class WanVideoOviCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "original_text_embeds": ("WANVIDEOTEXTEMBEDS", {"tooltip": "Base Wan text embeddings (positive + negative prompt) to extend with Ovi audio-branch CFG settings — connect from WanVideoTextEncode/Cached"}),
            "ovi_audio_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Separate CFG scale for the Ovi audio branch; higher = stronger audio-prompt adherence, 1.0 disables the audio uncond pass"}),
            },
            "optional": {
                "ovi_negative_text_embeds": ("WANVIDEOTEXTEMBEDS", {"tooltip": "Optional separate negative-prompt embeddings for the audio CFG pass; if omitted, the original positive embeddings are reused as the audio negative — connect from a second WanVideoTextEncode"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", )
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/Ovi"
    DESCRIPTION = "Adds Ovi negative text embeddings and audio CFG scale to the text embeddings dictionary"

    def process(self, original_text_embeds, ovi_audio_cfg, ovi_negative_text_embeds=None):
        negative_text_embeds = None
        if ovi_negative_text_embeds is not None:
            negative_text_embeds = ovi_negative_text_embeds.get("prompt_embeds", None)
        if negative_text_embeds is None:
            negative_text_embeds = original_text_embeds["prompt_embeds"]
            log.info("WanVideoOviCFG: Ovi negative text embeddings not provided, using original prompt embeddings as negative embeddings")
        else:
            log.info("WanVideoOviCFG: Using provided Ovi audio negative text embeddings")
        log.info("WanVideoOviCFG: negative text embedding shape: {}".format(negative_text_embeds[0].shape))

        prompt_embeds_dict_copy = original_text_embeds.copy()
        prompt_embeds_dict_copy.update({
                "ovi_negative_prompt_embeds": negative_text_embeds,
                "ovi_audio_cfg": ovi_audio_cfg,
            })
        return (prompt_embeds_dict_copy,)

NODE_CLASS_MAPPINGS = {
    "OviMMAudioVAELoader": OviMMAudioVAELoader,
    "WanVideoDecodeOviAudio": WanVideoDecodeOviAudio,
    "WanVideoEncodeOviAudio": WanVideoEncodeOviAudio,
    "WanVideoOviCFG": WanVideoOviCFG,
    "WanVideoAddOviAudioToLatents": WanVideoAddOviAudioToLatents,
    "WanVideoEmptyMMAudioLatents": WanVideoEmptyMMAudioLatents,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OviMMAudioVAELoader": "Ovi MMAudio VAE Loader",
    "WanVideoDecodeOviAudio": "WanVideo Decode Ovi Audio",
    "WanVideoEncodeOviAudio": "WanVideo Encode Ovi Audio",
    "WanVideoOviCFG": "WanVideo Ovi CFG",
    "WanVideoAddOviAudioToLatents": "WanVideo Add MMAudio To Latents",
    "WanVideoEmptyMMAudioLatents": "WanVideo Empty MMAudio Latents",
}