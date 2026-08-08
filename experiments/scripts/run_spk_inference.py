"""Run speaker/style consistency variants (Runs E / F) on the TED eval slice.

variant=spk      -> AudioVisualSpkModel
variant=spk_attn -> AudioVisualSpkAttnModel
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from tqdm import tqdm

MELGEN_CFG = dict(
    _name_="melgen", in_channels=80, out_channels=80,
    diffusion_step_embed_dim_in=128, diffusion_step_embed_dim_mid=512,
    diffusion_step_embed_dim_out=512, res_channels=512, skip_channels=512,
    num_res_layers=12, dilation_cycle=1, mel_upsample=[2, 2],
)


def _sampling_with_spk(net, diffusion_hyperparams, w_video, condition,
                      asr_guidance_net, w_asr, asr_start, guidance_text, tokenizer, decoder):
    """Mirror of inference_full_test_split.sampling with face_seq + audio_ref_emb threaded
    through both the conditional and unconditional `net(...)` calls. Attention (if present
    in the AudioVisualSpkAttnModel variant) is internal to the model."""
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    if asr_guidance_net is not None:
        text_tokens = torch.LongTensor(tokenizer.encode(guidance_text)).unsqueeze(0).cuda()

    mouthroi, face_image, face_seq, audio_ref_emb = condition
    x = torch.normal(0, 1, size=(mouthroi.shape[0], 80, mouthroi.shape[2] * 4)).cuda()
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()
            eps = net(x, mouthroi, face_image, face_seq, audio_ref_emb, diffusion_steps, cond_drop_prob=0)
            eps_uncond = net(x, mouthroi, face_image, face_seq, audio_ref_emb, diffusion_steps, cond_drop_prob=1)
            eps = (1 + w_video) * eps - w_video * eps_uncond

            if asr_guidance_net is not None and t <= asr_start:
                with torch.enable_grad():
                    length_input = torch.tensor([x.shape[2]]).cuda()
                    inputs = x.detach().requires_grad_(True), length_input
                    targets = text_tokens, torch.tensor([text_tokens.shape[1]]).cuda()
                    asr_guidance_net.device = torch.device("cuda")
                    batch_losses = asr_guidance_net.forward_model(inputs, diffusion_steps, targets, compute_metrics=True, verbose=0)[0]
                    asr_grad = torch.autograd.grad(batch_losses["loss"], inputs[0])[0]
                    asr_guidance_net.device = torch.device("cpu")
                grad_normaliser = torch.norm(eps / torch.sqrt(1 - Alpha_bar[t])) / torch.norm(asr_grad)
                eps = eps + torch.sqrt(1 - Alpha_bar[t]) * w_asr * grad_normaliser * asr_grad

            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * eps) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()
    return x


def run(variant: str, eval_split_path, spk_ckpt, videos_dir, mouthrois_dir, audios_dir,
        flow_dir, speaker_ref_dir, ref_clips_json, lipread_text_dir, out_audio_root,
        hifi_gan_config="hifi_gan/config.json", hifi_gan_ckpt="hifi_gan/g_02400000",
        w_video=2.0, w_asr=1.5, asr_start=270, sampling_rate=16000, max_clips=None):
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    from experiments.scripts.run_baseline_inference import _link_checkpoints
    _link_checkpoints()

    from dataloaders.dataset_lipvoicer import LipVoicerDataset
    from dataloaders.stft import denormalise_mel
    from models.model_builder import ModelBuilder
    from models.audiovisual_spk_model import AudioVisualSpkModel, AudioVisualSpkAttnModel
    from models.speaker_style import SpeakerStyleFusion, VisualStyleEncoder
    from models.temporal_attention import TemporalAttention
    import ASR.asr_models as asr_models
    from hifi_gan.generator import Generator as Vocoder
    from hifi_gan import utils as vocoder_utils
    from hifi_gan.env import AttrDict
    from utils import calc_diffusion_hyperparams
    from experiments.checkpoint_compat import load as compat_load

    eval_ids = {x.strip().removeprefix("main/") for x in Path(eval_split_path).read_text().splitlines() if x.strip()}
    print(f"eval split: {len(eval_ids)} clip IDs")

    dh = calc_diffusion_hyperparams(T=400, beta_0=0.0001, beta_T=0.02, beta=None, fast=True)

    builder = ModelBuilder()
    backbones = (builder.build_lipreadingnet(), builder.build_facial(fc_out=128, with_fc=True),
                 builder.build_diffwave_model(OmegaConf.create(MELGEN_CFG)))
    if variant == "spk":
        net = AudioVisualSpkModel(backbones,
                                   speaker_style_fusion=SpeakerStyleFusion(),
                                   visual_style_encoder=VisualStyleEncoder()).cuda().eval()
    elif variant == "spk_attn":
        net = AudioVisualSpkAttnModel(backbones,
                                       speaker_style_fusion=SpeakerStyleFusion(),
                                       visual_style_encoder=VisualStyleEncoder(),
                                       temporal_attention=TemporalAttention()).cuda().eval()
    else:
        raise ValueError(f"unknown variant: {variant}")

    print(f"loading {variant} ckpt from {spk_ckpt}")
    state = compat_load(spk_ckpt, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    net.load_state_dict(state)
    print(f"{variant} model loaded")

    asr_guidance_net, tokenizer, decoder = asr_models.get_models("LRS2")

    with open(hifi_gan_config) as f:
        h = AttrDict(json.loads(f.read()))
    vocoder = Vocoder(h).cuda()
    vocoder.load_state_dict(vocoder_utils.load_checkpoint(hifi_gan_ckpt, "cuda")["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()

    alias = "/tmp/LRS2_videos"
    if not os.path.lexists(alias):
        os.symlink(videos_dir, alias)
    dataset = LipVoicerDataset("test", videos_dir=alias, mouthrois_dir=mouthrois_dir,
                               audios_dir=audios_dir, sampling_rate=sampling_rate,
                               videos_window_size=25, audio_stft_hop=160,
                               flow_dir=flow_dir, speaker_ref_dir=speaker_ref_dir,
                               ref_clips_json=ref_clips_json)
    print(f"dataset size: {len(dataset)}")

    out_root = Path(out_audio_root)
    out_root.mkdir(parents=True, exist_ok=True)
    n_done = 0

    for i in tqdm(range(len(dataset)), desc=f"{variant} inference"):
        try:
            _mel, _audio, mouthroi, face_image, _flow, face_seq, audio_ref_emb, gt_text, video_id = dataset[i]
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  dataset[{i}] failed: {e}")
            continue

        if video_id not in eval_ids:
            continue

        _, fh, fw = face_image.shape
        if fh != fw:
            side = min(fh, fw)
            face_image = TF.center_crop(face_image, [side, side])
            face_image = TF.resize(face_image, [224, 224], antialias=True)

        out_wav = out_root / f"{video_id}.wav"
        if out_wav.exists() and out_wav.stat().st_size > 0:
            n_done += 1
            if max_clips and n_done >= max_clips:
                break
            continue

        text_path = Path(lipread_text_dir) / f"{video_id}.txt"
        text = text_path.read_text().strip() if text_path.exists() else gt_text
        text = text or gt_text

        mouthroi = mouthroi.unsqueeze(0).cuda()
        face_image = face_image.unsqueeze(0).cuda()
        face_seq = face_seq.unsqueeze(0).cuda()
        audio_ref_emb = audio_ref_emb.unsqueeze(0).cuda()

        try:
            mel = _sampling_with_spk(net, dh, w_video, (mouthroi, face_image, face_seq, audio_ref_emb),
                                     asr_guidance_net, w_asr, asr_start, text, tokenizer, decoder)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  sampling failed for {video_id}: {e}")
            continue

        audio = vocoder(denormalise_mel(mel)).squeeze()
        audio = audio / 1.1 / audio.abs().max()
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), audio.detach().cpu().numpy(), sampling_rate)
        n_done += 1
        if max_clips and n_done >= max_clips:
            break

    print(f"\ngenerated {n_done} clips. Output: {out_root}")
