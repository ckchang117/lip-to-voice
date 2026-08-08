"""Run LipVoicer's pretrained MelGen on the TED eval_200utt slice -> baseline audio.

Reuses LipVoicer's `sampling()` function (the actual diffusion sampling loop)
verbatim, but wires in:
  - Our TED-on-LRS2-layout dataset paths
  - Eval-subset filtering (only run on eval_200utt.txt IDs)
  - Output to /vol/baseline_audio/ted/<talk>/<sent>.wav

The downstream eval harness (experiments/scripts/eval_baseline.py) is agnostic to
where the audio came from, so the same `eval_baseline` Modal function will
score these outputs.

Must run with CWD = /root/repo so LipVoicer's hardcoded relative paths (e.g.
'hifi_gan/config.json') resolve.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def _link_checkpoints(repo_root: str = "/root/repo", ckpt_root: str = "/vol/checkpoints/lipvoicer") -> None:
    """File-level symlinks for LipVoicer's pretrained files into the repo tree."""
    from pathlib import Path
    repo = Path(repo_root)
    ckpt = Path(ckpt_root)
    targets = [
        ("exp/LRS2/wnet_h512_d12_T400_betaT0.02/checkpoint/1000000.pkl", "file"),
        ("ASR/callbacks/LRS23/AO/EffConfCTC/checkpoints_ft_lrs2.ckpt", "file"),
        ("ASR/callbacks/LRS23/LM/GPT-Small/checkpoints_epoch_10_step_2860.ckpt", "file"),
        ("ASR/media/tokenizerbpe256.model", "file"),
        ("ASR/media/tokenizerbpe1024.model", "file"),
        ("ASR/media/6gram_lrs23.arpa", "file"),
        ("hifi_gan/g_02400000", "file"),
        ("mouthroi_processing/benchmarks/LRS3/models/LRS3_V_WER19.1", "dir"),
        ("mouthroi_processing/benchmarks/LRS3/language_models/lm_en_subword", "dir"),
    ]
    for rel, kind in targets:
        src = ckpt / rel
        dst = repo / rel
        if not src.exists():
            print(f"  [skip] missing in checkpoints: {src}")
            continue
        if dst.is_symlink() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src)
            print(f"  linked {dst} -> {src}")
        except OSError as e:
            print(f"  WARN symlink failed {dst}: {e}")


def _setup_repo() -> None:
    if "/root/repo" not in sys.path:
        sys.path.insert(0, "/root/repo")
    os.chdir("/root/repo")
    _link_checkpoints()


def run(
    eval_split_path: str,
    melgen_ckpt: str,
    videos_dir: str,
    mouthrois_dir: str,
    audios_dir: str,
    lipread_text_dir: str,
    out_audio_root: str,
    hifi_gan_config: str = "hifi_gan/config.json",
    hifi_gan_ckpt: str = "hifi_gan/g_02400000",
    w_video: float = 2.0,
    w_asr: float = 1.5,
    asr_start: int = 270,
    sampling_rate: int = 16000,
    max_clips: int | None = None,
) -> None:
    _setup_repo()

    # Imports must happen after sys.path is set
    from dataloaders.dataset_lipvoicer import LipVoicerDataset
    from dataloaders.stft import denormalise_mel
    from models.model_builder import ModelBuilder
    from models.audiovisual_model import AudioVisualModel
    import ASR.asr_models as asr_models
    from hifi_gan.generator import Generator as Vocoder
    from hifi_gan import utils as vocoder_utils
    from hifi_gan.env import AttrDict
    from utils import calc_diffusion_hyperparams, print_size
    from inference_full_test_split import sampling

    # Load eval split IDs into a set for fast lookup. Format: "main/<talk>/<sent>"
    eval_ids_raw = [
        x.strip() for x in Path(eval_split_path).read_text().splitlines() if x.strip()
    ]
    # LipVoicerDataset prepends "main/" then ".npz" - but the IDs it produces
    # for video_id (see line 99 of inference_full_test_split.py) are
    # "<talk>/<sent>" (parent + stem). Normalise both forms.
    eval_id_set = set()
    for x in eval_ids_raw:
        x = x.removeprefix("main/")
        eval_id_set.add(x)
    print(f"eval split: {len(eval_id_set)} clip IDs")

    # Build the diffusion hyperparams (matches configs/config.yaml diffusion: section).
    diffusion_hyperparams = calc_diffusion_hyperparams(
        T=400, beta_0=0.0001, beta_T=0.02, beta=None, fast=True
    )

    # Build the MelGen network
    melgen_cfg = OmegaConf.create({
        "_name_": "melgen",
        "in_channels": 80,
        "out_channels": 80,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "res_channels": 512,
        "skip_channels": 512,
        "num_res_layers": 12,
        "dilation_cycle": 1,
        "mel_upsample": [2, 2],
    })

    builder = ModelBuilder()
    net_lipreading = builder.build_lipreadingnet()
    net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_diffwave_model(melgen_cfg)
    net = AudioVisualModel((net_lipreading, net_facial, net_diffwave)).cuda()
    net.eval()
    print_size(net)

    # Load MelGen checkpoint via our compat shim (handles LipVoicer pickle issue #7)
    sys.path.insert(0, "/root/repo")  # ensure experiments.checkpoint_compat is importable
    from experiments.checkpoint_compat import load as compat_load

    print(f"loading MelGen ckpt from {melgen_ckpt}")
    state = compat_load(melgen_ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        net.load_state_dict(state["model_state_dict"])
    else:
        net.load_state_dict(state)
    print("MelGen loaded")

    # Force ds_name=LRS2 (we mimic LRS2 layout - main/ subdir, no test/ subdir).
    ds_name = "LRS2"

    # ASR for classifier guidance
    print("loading ASR + tokenizer + decoder")
    asr_guidance_net, tokenizer, decoder = asr_models.get_models(ds_name)

    # HiFi-GAN vocoder
    print(f"loading HiFi-GAN from {hifi_gan_ckpt}")
    with open(hifi_gan_config) as f:
        h = AttrDict(json.loads(f.read()))
    vocoder = Vocoder(h).cuda()
    state_g = vocoder_utils.load_checkpoint(hifi_gan_ckpt, "cuda")
    vocoder.load_state_dict(state_g["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()

    # LipVoicerDataset detects LRS2 vs LRS3 via case-sensitive `"LRS2" in videos_dir`.
    # Our volume path uses lowercase "lrs2" - symlink an alias with uppercase.
    alias_dir = "/tmp/LRS2_videos"
    if not os.path.lexists(alias_dir):
        os.symlink(videos_dir, alias_dir)
    dataset_cfg = dict(
        videos_dir=alias_dir,
        mouthrois_dir=mouthrois_dir,
        audios_dir=audios_dir,
        sampling_rate=sampling_rate,
        videos_window_size=25,
        audio_stft_hop=160,
    )
    dataset = LipVoicerDataset("test", **dataset_cfg)
    print(f"dataset size: {len(dataset)}")

    out_root = Path(out_audio_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skipped = 0

    import torchvision.transforms.functional as TF

    for i in tqdm(range(len(dataset)), desc="diffusion inference"):
        try:
            gt_melspec, _gt_audio, mouthroi, face_image, gt_text, video_id = dataset[i]
        except Exception as e:
            tqdm.write(f"  dataset[{i}] failed: {e}")
            continue

        # LipVoicerDataset.get_face_image_transform() uses Resize(224) which only
        # constrains the SHORTER side. For our 16:9 TED frames that produces
        # (3, 224, ~398) instead of (3, 224, 224). Center-crop to square, then
        # resize to the model's expected (224, 224).
        _, fh, fw = face_image.shape
        if fh != fw:
            side = min(fh, fw)
            face_image = TF.center_crop(face_image, [side, side])
            face_image = TF.resize(face_image, [224, 224], antialias=True)

        # Filter to eval split
        if video_id not in eval_id_set:
            continue

        out_wav = out_root / f"{video_id}.wav"
        if out_wav.exists() and out_wav.stat().st_size > 0:
            n_skipped += 1
            n_done += 1
            if max_clips and n_done >= max_clips:
                break
            continue

        # Read pre-computed lipread text
        text_path = Path(lipread_text_dir) / f"{video_id}.txt"
        if not text_path.exists():
            tqdm.write(f"  WARN no lipread text for {video_id}, using GT (cheating)")
            text = gt_text
        else:
            text = text_path.read_text().strip() or gt_text

        mouthroi = mouthroi.unsqueeze(0).cuda()
        face_image = face_image.unsqueeze(0).cuda()

        # Run diffusion sampling
        try:
            mel = sampling(
                net,
                diffusion_hyperparams,
                w_video,
                condition=(mouthroi, face_image),
                asr_guidance_net=asr_guidance_net,
                w_asr=w_asr,
                asr_start=asr_start,
                guidance_text=text,
                tokenizer=tokenizer,
                decoder=decoder,
            )
        except Exception as e:
            tqdm.write(f"  sampling failed for {video_id}: {e}")
            continue

        mel = denormalise_mel(mel)
        audio = vocoder(mel)
        audio = audio.squeeze()
        audio = audio / 1.1 / audio.abs().max()
        audio_np = audio.detach().cpu().numpy()

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), audio_np, sampling_rate)
        n_done += 1

        if max_clips and n_done >= max_clips:
            break

    print(f"\ngenerated {n_done} clips ({n_skipped} cached). Output: {out_root}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--eval_split", required=True)
    p.add_argument("--melgen_ckpt", required=True)
    p.add_argument("--videos_dir", required=True)
    p.add_argument("--mouthrois_dir", required=True)
    p.add_argument("--audios_dir", required=True)
    p.add_argument("--lipread_text_dir", required=True)
    p.add_argument("--out_audio_root", required=True)
    p.add_argument("--hifi_gan_config", default="hifi_gan/config.json")
    p.add_argument("--hifi_gan_ckpt", default="hifi_gan/g_02400000")
    p.add_argument("--w_video", type=float, default=2.0)
    p.add_argument("--w_asr", type=float, default=1.5)
    p.add_argument("--asr_start", type=int, default=270)
    p.add_argument("--max_clips", type=int, default=None)
    args = p.parse_args()
    run(
        eval_split_path=args.eval_split,
        melgen_ckpt=args.melgen_ckpt,
        videos_dir=args.videos_dir,
        mouthrois_dir=args.mouthrois_dir,
        audios_dir=args.audios_dir,
        lipread_text_dir=args.lipread_text_dir,
        out_audio_root=args.out_audio_root,
        hifi_gan_config=args.hifi_gan_config,
        hifi_gan_ckpt=args.hifi_gan_ckpt,
        w_video=args.w_video,
        w_asr=args.w_asr,
        asr_start=args.asr_start,
        max_clips=args.max_clips,
    )
