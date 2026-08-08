"""Cross-speaker swap test - GENERATION PHASE.

Generates audio for two configurations per eval clip:
  1. SWAP: speaker A's silent video + speaker B's audio reference -> "what if B were speaking A's lines?"
  2. NONSWAP: speaker A's silent video + speaker A's audio reference (sanity / calibration).

Writes the audio files + a metadata JSON listing (clip_A, talk_B, ref_clip_paths). The
scoring phase (experiments/scripts/swap_test_score.py) runs in SPEAKERSIM_IMAGE with
speechbrain, reads this metadata, runs ECAPA, and produces the final swap-test JSON.

The split avoids LIPVOICER_IMAGE having to import speechbrain (which transitively imports
torchaudio.io.StreamReader -> requires FFmpeg dev libraries not in our base image).
"""

from __future__ import annotations

import json
import os
import random
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


def _load_precomputed(emb_root: Path, talk: str, clip: str):
    p = emb_root / "main" / talk / f"{clip}.npy"
    if not p.exists():
        return None
    return torch.from_numpy(np.load(str(p))).float()


def _sampling(net, dh, w_video, condition, asr_guidance_net, w_asr, asr_start, text, tokenizer):
    from experiments.scripts.run_spk_inference import _sampling_with_spk
    return _sampling_with_spk(net, dh, w_video, condition, asr_guidance_net, w_asr,
                              asr_start, text, tokenizer, None)


def run(variant: str, eval_split_path, spk_ckpt, videos_dir, mouthrois_dir, audios_dir,
        flow_dir, speaker_ref_dir, ref_clips_json, lipread_text_dir, out_audio_root,
        out_json_path, n_pairs: int = 50, seed: int = 1234,
        hifi_gan_config="hifi_gan/config.json",
        hifi_gan_ckpt="hifi_gan/g_02400000",
        w_video=2.0, w_asr=1.5, asr_start=270, sampling_rate=16000):
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    from experiments.scripts.run_baseline_inference import _link_checkpoints
    _link_checkpoints()

    from dataloaders.dataset_lipvoicer import LipVoicerDataset
    from dataloaders.stft import denormalise_mel
    from models.model_builder import ModelBuilder
    from models.audiovisual_spk_model import AudioVisualSpkModel, AudioVisualSpkAttnModel
    from models.audiovisual_spk_film_model import (
        AudioVisualSpkFilmModel, AudioVisualSpkAttnFilmModel,
    )
    from models.speaker_style import SpeakerStyleFusion, VisualStyleEncoder
    from models.speaker_style_film import SpeakerStyleFiLM, LayerFiLMBank
    from models.temporal_attention import TemporalAttention
    import ASR.asr_models as asr_models
    from hifi_gan.generator import Generator as Vocoder
    from hifi_gan import utils as vocoder_utils
    from hifi_gan.env import AttrDict
    from utils import calc_diffusion_hyperparams
    from experiments.checkpoint_compat import load as compat_load

    rng = random.Random(seed)

    eval_ids = [x.strip().removeprefix("main/") for x in Path(eval_split_path).read_text().splitlines() if x.strip()]
    refs = json.loads(Path(ref_clips_json).read_text())
    emb_root = Path(speaker_ref_dir)

    # Build talk pool excluding the talks present in the eval set (so swap-target speakers are clearly different).
    eval_talks = {vid.split("/")[0] for vid in eval_ids}
    other_talks = [t for t in refs.keys() if t not in eval_talks and refs[t]]
    if not other_talks:
        raise SystemExit("no other talks available for swap test")

    # Build the model (same code paths as run_spk_inference)
    dh = calc_diffusion_hyperparams(T=400, beta_0=0.0001, beta_T=0.02, beta=None, fast=True)
    builder = ModelBuilder()
    backbones = (builder.build_lipreadingnet(), builder.build_facial(fc_out=128, with_fc=True),
                 builder.build_diffwave_model(OmegaConf.create(MELGEN_CFG)))
    if variant == "spk":
        net = AudioVisualSpkModel(backbones, speaker_style_fusion=SpeakerStyleFusion(),
                                   visual_style_encoder=VisualStyleEncoder()).cuda().eval()
    elif variant == "spk_attn":
        net = AudioVisualSpkAttnModel(backbones, speaker_style_fusion=SpeakerStyleFusion(),
                                       visual_style_encoder=VisualStyleEncoder(),
                                       temporal_attention=TemporalAttention()).cuda().eval()
    elif variant == "g":
        net = AudioVisualSpkFilmModel(backbones, speaker_style_fusion=SpeakerStyleFiLM(),
                                       visual_style_encoder=VisualStyleEncoder(),
                                       layer_film_bank=LayerFiLMBank()).cuda().eval()
    elif variant == "h":
        net = AudioVisualSpkAttnFilmModel(backbones, speaker_style_fusion=SpeakerStyleFiLM(),
                                           visual_style_encoder=VisualStyleEncoder(),
                                           layer_film_bank=LayerFiLMBank(),
                                           temporal_attention=TemporalAttention()).cuda().eval()
    else:
        raise ValueError(f"unknown variant: {variant}")
    state = compat_load(spk_ckpt, map_location="cpu")
    state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    net.load_state_dict(state)

    asr_guidance_net, tokenizer, _decoder = asr_models.get_models("LRS2")
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
    print(f"swap test for variant={variant} | dataset size={len(dataset)} | n_pairs={n_pairs}")

    out_root = Path(out_audio_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build a dict from video_id -> dataset index for fast lookup
    id_to_idx = {}
    for idx in range(len(dataset)):
        try:
            sample = dataset.moutroi_files[idx]
        except Exception:
            continue
        # mouthroi filename -> video_id
        from pathlib import Path as _P
        pf = _P(sample)
        vid = "/".join([pf.parts[-2], pf.stem])
        id_to_idx[vid] = idx

    # Pick n_pairs from eval; for each, pick a random other-speaker as B (target voice)
    eval_targets = [vid for vid in eval_ids if vid in id_to_idx]
    rng.shuffle(eval_targets)
    eval_targets = eval_targets[:n_pairs]

    per_pair = {}
    for vid_A in tqdm(eval_targets, desc="swap-gen"):
        talk_A, clip_A = vid_A.split("/")
        talk_B = rng.choice(other_talks)
        ref_B_clip = refs[talk_B][0][0]
        ref_B_emb = _load_precomputed(emb_root, talk_B, ref_B_clip)
        if ref_B_emb is None:
            continue
        ref_A_clip = refs[talk_A][0][0]
        ref_A_emb = _load_precomputed(emb_root, talk_A, ref_A_clip)
        if ref_A_emb is None:
            continue

        idx = id_to_idx[vid_A]
        _mel, _audio, mouthroi, face_image, _flow, face_seq, _ref_emb_A, gt_text, video_id = dataset[idx]
        assert video_id == vid_A

        _, fh, fw = face_image.shape
        if fh != fw:
            side = min(fh, fw)
            face_image = TF.center_crop(face_image, [side, side])
            face_image = TF.resize(face_image, [224, 224], antialias=True)

        text_path = Path(lipread_text_dir) / f"{vid_A}.txt"
        text = text_path.read_text().strip() if text_path.exists() else gt_text
        text = text or gt_text

        mouthroi_g = mouthroi.unsqueeze(0).cuda()
        face_image_g = face_image.unsqueeze(0).cuda()
        face_seq_g = face_seq.unsqueeze(0).cuda()

        # 1) SWAP: A's video + B's audio reference
        swap_wav = out_root / f"{vid_A.replace('/', '__')}__swap_{talk_B}.wav"
        if not swap_wav.exists():
            try:
                mel = _sampling(net, dh, w_video, (mouthroi_g, face_image_g, face_seq_g, ref_B_emb.unsqueeze(0).cuda()),
                                asr_guidance_net, w_asr, asr_start, text, tokenizer)
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"  swap sampling failed for {vid_A}: {e}")
                continue
            audio = vocoder(denormalise_mel(mel)).squeeze()
            audio = audio / 1.1 / audio.abs().max()
            sf.write(str(swap_wav), audio.detach().cpu().numpy(), sampling_rate)

        # 2) NONSWAP: A's video + A's audio reference (calibration)
        ns_wav = out_root / f"{vid_A.replace('/', '__')}__nonswap.wav"
        if not ns_wav.exists():
            try:
                mel = _sampling(net, dh, w_video, (mouthroi_g, face_image_g, face_seq_g, ref_A_emb.unsqueeze(0).cuda()),
                                asr_guidance_net, w_asr, asr_start, text, tokenizer)
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"  nonswap sampling failed for {vid_A}: {e}")
                continue
            audio = vocoder(denormalise_mel(mel)).squeeze()
            audio = audio / 1.1 / audio.abs().max()
            sf.write(str(ns_wav), audio.detach().cpu().numpy(), sampling_rate)

        per_pair[vid_A] = {
            "talk_A": talk_A, "talk_B": talk_B,
            "swap_wav": str(swap_wav.relative_to(out_root.parent.parent)) if out_root.parent.parent in swap_wav.parents else str(swap_wav),
            "nonswap_wav": str(ns_wav.relative_to(out_root.parent.parent)) if out_root.parent.parent in ns_wav.parents else str(ns_wav),
            "ref_A_clip": ref_A_clip,
            "ref_B_clip": ref_B_clip,
        }

    payload = {"variant": variant, "n_pairs": len(per_pair), "per_pair": per_pair,
               "audio_root": str(out_root)}
    out_json = Path(out_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"saved metadata -> {out_json}")
    print(f"generated {len(per_pair)} swap + nonswap pairs in {out_root}")
    print(f"next: run swap_test_score (SPEAKERSIM_IMAGE) to compute ECAPA similarities")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["spk", "spk_attn"])
    for a in ("eval_split_path", "spk_ckpt", "videos_dir", "mouthrois_dir", "audios_dir",
              "flow_dir", "speaker_ref_dir", "ref_clips_json", "lipread_text_dir",
              "out_audio_root", "out_json_path"):
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--n_pairs", type=int, default=50)
    args = p.parse_args()
    run(args.variant, args.eval_split_path, args.spk_ckpt, args.videos_dir, args.mouthrois_dir,
        args.audios_dir, args.flow_dir, args.speaker_ref_dir, args.ref_clips_json,
        args.lipread_text_dir, args.out_audio_root, args.out_json_path, n_pairs=args.n_pairs)
