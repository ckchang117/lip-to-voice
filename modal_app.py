"""Modal entry point: data pipeline, training, inference, and evaluation.

Runs everything that needs a GPU or large network egress on Modal, against a
single persistent volume at /vol; nothing computes locally.

Usage (after `pip install modal && modal token new`):

    modal run modal_app.py --step <function> [--variant <v>] [--eval-split <f>]

e.g. `modal run modal_app.py --step train_ablation --variant c`. Any function
below is a valid step. Each is small, idempotent, and resumable: rerunning a
stage after a partial run or crash skips completed work.
"""

from __future__ import annotations

from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Image
# --------------------------------------------------------------------------
# Two images: a light one for downloads (no GPU, no torch) and a GPU image
# for evaluation (Whisper + wav2vec2 + SyncNet). The GPU image inherits
# from the light one to share apt + base pip layers.

# Base layer shared across light + segment images - apt and the common
# pip deps, but NO add_local_* (which must be the final step in any image).
_BASE_LIGHT = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "curl")
    .pip_install(
        "gdown==5.2.0",
        "huggingface-hub==0.24.0",
        "datasets==2.20.0",
        "pyarrow==17.0.0",
        "tqdm==4.66.4",
        "requests==2.32.3",
        # TED pipeline additions
        "yt-dlp==2024.8.6",
        "webvtt-py==0.5.1",
        "beautifulsoup4==4.12.3",
        "lxml==5.3.0",
    )
)

LIGHT_IMAGE = _BASE_LIGHT.add_local_python_source("experiments")

# Image for the segment_ted stage. Now transcripts come from TED's GraphQL
# (cached in talk.transcript.json during download), so we no longer need
# Whisper / torch - just mediapipe for the single-face filter + ffmpeg.
SEGMENT_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "curl")
    .pip_install(
        "mediapipe==0.10.14",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "Pillow==10.4.0",
        "requests==2.32.3",
        "tqdm==4.66.4",
    )
    .add_local_python_source("experiments")
)

# Heavy image with LipVoicer's full preprocessing + inference stack.
# Uses LipVoicer's pinned torch 1.13.0 + CUDA 11.7 so the released checkpoints
# load cleanly. Includes the repo itself at /root/repo so we can import the
# project's existing modules (mouthroi_processing, dataloaders, models, hifi_gan, ASR).
LIPVOICER_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "git-lfs", "curl", "build-essential", "cmake", "wget")
    .run_commands("git lfs install")
    .pip_install(
        "torch==1.13.0",
        "torchaudio==0.13.0",
        "torchvision==0.14.0",
        index_url="https://download.pytorch.org/whl/cu117",
    )
    .pip_install(
        # From LipVoicer's requirements.txt - bumped av to 12.x for prebuilt wheels.
        # LipVoicer only uses av.open / AudioResampler / AVError, all stable across versions.
        "av==12.3.0",
        "editdistance==0.8.1",
        "gdown==5.2.0",
        "hydra-core==1.3.2",
        "jiwer==3.0.4",
        "librosa==0.10.2",
        "matplotlib==3.6.2",
        "numpy==1.23.4",
        "omegaconf==2.3.0",
        "opencv-contrib-python==4.8.0.76",
        "Pillow==10.4.0",
        "scikit-image==0.21.0",
        "scipy==1.10.1",
        "sentencepiece==0.1.99",
        "six==1.16.0",
        "soundfile==0.12.1",
        "torch-complex==0.4.3",
        "tqdm==4.66.4",
        "requests==2.32.3",
        # Required by LipVoicer's ASR module (`ASR/nnet/model.py` imports
        # `torch.utils.tensorboard.SummaryWriter` at module load).
        "tensorboard==2.13.0",
        # Mouth-ROI extraction deps. We use mediapipe (not retinaface) because
        # ibug.face_detection's GitHub LFS budget is exhausted and downloads fail.
        # The InferencePipeline supports detector="mediapipe" as a built-in alternative.
        "mediapipe==0.10.14",
    )
    .run_commands(
        # ctcdecode for ASR beam search (used in classifier-guidance)
        "pip install --no-build-isolation git+https://github.com/WayenVan/ctcdecode.git",
    )
    # Bring the LipVoicer-rooted code into the container so we can import its
    # modules. Excludes large dirs that the container doesn't need.
    .add_local_dir(
        local_path=".",
        remote_path="/root/repo",
        ignore=["**/.git/**", "**/__pycache__/**", "**/.modal/**", "**/.venv/**", "**/*.pyc"],
    )
)

GPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "curl")
    .pip_install(
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "torchvision==0.17.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "gdown==5.2.0",
        "huggingface-hub==0.24.0",
        "datasets==2.20.0",
        "pyarrow==17.0.0",
        "tqdm==4.66.4",
        "requests==2.32.3",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "soundfile==0.12.1",
        "librosa==0.10.2",
        "scipy==1.13.1",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "scikit-image==0.24.0",
        "av==12.3.0",
        "jiwer==3.0.4",
        "pystoi==0.4.1",
        "matplotlib==3.9.2",
        # DNSMOS via speechmos - wraps Microsoft's official ONNX models.
        "speechmos==0.0.1",
        "onnxruntime==1.18.1",
    )
    .add_local_python_source("experiments")
)

# ECAPA-TDNN speaker-embedding precompute + speaker-similarity / swap-test eval.
# Kept separate from LIPVOICER_IMAGE so the heavier training image doesn't carry
# speechbrain at runtime - we only read precomputed .npy embeddings during training.
SPEAKERSIM_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "curl")
    .pip_install(
        "torch==2.2.0",
        "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "speechbrain==1.0.0",
        "huggingface-hub==0.24.0",
        "soundfile==0.12.1",
        "tqdm==4.66.4",
        "numpy==1.26.4",
    )
    .add_local_python_source("experiments")
)

# --------------------------------------------------------------------------
# Volume + secrets + app
# --------------------------------------------------------------------------
vol = modal.Volume.from_name("lip-to-voice", create_if_missing=True)
VOL_MOUNT = "/vol"

# HuggingFace token, needed for the gated `mahmoodanaam/lrs2` dataset.
# Create it once on your Mac with:
#     modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
# (Generate the token at https://huggingface.co/settings/tokens.)
hf_secret = modal.Secret.from_name("huggingface")

app = modal.App("lip-to-voice", image=LIGHT_IMAGE)

# --------------------------------------------------------------------------
# Volume layout (created on first use)
# --------------------------------------------------------------------------
# /vol/checkpoints/lipvoicer/         LipVoicer's released pretrained checkpoints (gdrive)
# /vol/baseline_audio/lrs2/           LipVoicer's pre-generated LRS2 test audio (gdrive)
# /vol/lipread_text/lrs2_test/        LipVoicer's pre-generated test-set lipread predictions (gdrive)
# /vol/lrs2_raw/                      Test videos extracted from mahmoodanaam/lrs2 parquet
#   ├── main/<video_id>.mp4
#   └── main/<video_id>.txt           (LRS2-style "Text:  ..." transcript)
# /vol/lrs2_proc/mouth_roi/main/<video_id>.npz   (created on demand for SyncNet)
# /vol/eval/                          Pinned SyncNet / Whisper / wav2vec2 weights
# /vol/results/                       JSON output of eval_baseline

# --------------------------------------------------------------------------
# Step 1: Download LipVoicer's pretrained checkpoints
# --------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
    cpu=2.0,
    memory=4096,
)
def download_pretrained_checkpoints():
    """Mirror LipVoicer's download_checkpoints.py into /vol/checkpoints/lipvoicer/.

    We only download the LRS2 variants (we're not using LRS3). Idempotent: skips
    files that already exist with non-zero size.
    """
    import os
    from pathlib import Path
    from zipfile import ZipFile

    import gdown

    # Subset of LipVoicer's download_checkpoints.py - LRS2-relevant only.
    repo_files = {
        "MelGen_LRS2": {
            "gdrive_id": "1gTIpxaMx31ZUhPd2jW8Bt8u4QknieO31",
            "dir_path": "exp/LRS2/wnet_h512_d12_T400_betaT0.02/checkpoint",
            "filename": "1000000.pkl",
        },
        "ASR_LRS2": {
            "gdrive_id": "1adeCf4NzhshJVU-JndlKpC34rRwJOQ2B",
            "dir_path": "ASR/callbacks/LRS23/AO/EffConfCTC",
            "filename": "checkpoints_ft_lrs2.ckpt",
        },
        "TransformerLM": {
            "gdrive_id": "1PSo4ZQIZPWEI_S5LHkJBo0gYhQpWzRnh",
            "dir_path": "ASR/callbacks/LRS23/LM/GPT-Small",
            "filename": "checkpoints_epoch_10_step_2860.ckpt",
        },
        "Tokenizer": {
            "gdrive_id": "1u3U3aHaTWvR_NTftkUGv1JXkxpX1pkOL",
            "dir_path": "ASR/media",
            "filename": "tokenizerbpe256.model",
        },
        "TokenizerLM": {
            "gdrive_id": "1zKp376kItVhceTFSi2_-EMG3oeYbSC0U",
            "dir_path": "ASR/media",
            "filename": "tokenizerbpe1024.model",
        },
        "6gramLM": {
            "gdrive_id": "1l71jUmRdQMFO2AVezxweENpZgdvL7TyD",
            "dir_path": "ASR/media",
            "filename": "6gram_lrs23.arpa",
        },
        "HiFi-GAN": {
            "gdrive_id": "1h0gcgifwe5HVM76rlREHj1daBNItWh7e",
            "dir_path": "hifi_gan",
            "filename": "g_02400000",
        },
        "LipReader": {
            "gdrive_id": "1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6",
            "dir_path": "mouthroi_processing/benchmarks/LRS3/models",
            "filename": "LRS3_V_WER19.1.zip",
        },
        "LipReaderLM": {
            "gdrive_id": "1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv",
            "dir_path": "mouthroi_processing/benchmarks/LRS3/language_models",
            "filename": "lm_en_subword.zip",
        },
    }

    root = Path(VOL_MOUNT) / "checkpoints" / "lipvoicer"
    root.mkdir(parents=True, exist_ok=True)

    for key, value in repo_files.items():
        out_dir = root / value["dir_path"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / value["filename"]

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[skip] {key}: already at {out_path}")
            continue

        url = f"https://drive.google.com/uc?id={value['gdrive_id']}"
        print(f"[download] {key}: {url}  ->  {out_path}")
        gdown.download(url, str(out_path), quiet=False)

        if out_path.suffix == ".zip":
            print(f"[unzip] {out_path}")
            with ZipFile(out_path) as z:
                z.extractall(out_dir)

    vol.commit()
    print("download_pretrained_checkpoints: done")


# --------------------------------------------------------------------------
# Step 2: Download LipVoicer's pre-generated LRS2 baseline audio
# --------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
    cpu=2.0,
    memory=4096,
)
def download_lipvoicer_baseline_audio():
    """Download the LipVoicer authors' pre-generated LRS2 test audio.

    URL from README.md: https://drive.google.com/uc?id=15BgeaU-j4B-o9WfP-d-BSnnjslAxOHkt

    This lets us reproduce the LipVoicer LRS2 baseline numbers without running
    diffusion sampling ourselves - we just need to evaluate metrics on the
    audio they produced.
    """
    import zipfile
    from pathlib import Path

    import gdown

    out_dir = Path(VOL_MOUNT) / "baseline_audio" / "lrs2"
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = out_dir / "lipvoicer_lrs2.zip"
    if not (out_dir / "EXTRACTED").exists():
        if not archive.exists() or archive.stat().st_size == 0:
            url = "https://drive.google.com/uc?id=15BgeaU-j4B-o9WfP-d-BSnnjslAxOHkt"
            print(f"[download] {url}  ->  {archive}")
            gdown.download(url, str(archive), quiet=False)

        print(f"[unzip] {archive}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
        (out_dir / "EXTRACTED").touch()
    else:
        print(f"[skip] baseline audio already extracted under {out_dir}")

    # Inventory: log the first few entries so we can see the file layout.
    print("Top-level contents:")
    for p in sorted(out_dir.glob("*"))[:20]:
        print(f"  {p.relative_to(out_dir)}")

    vol.commit()
    print("download_lipvoicer_baseline_audio: done")


# --------------------------------------------------------------------------
# Step 3: Download LipVoicer's pre-generated lipread predictions
# --------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
    cpu=2.0,
    memory=4096,
)
def download_lipread_predictions():
    """Download the per-clip lip-reading text predictions for the LRS2 test set.

    URL from README: https://drive.google.com/uc?id=1T4ZFGvW9643844BaOnJyo78-KN83h8Yf

    These are only required if we want to *regenerate* LipVoicer's baseline audio
    ourselves (since the diffusion needs text guidance). For pure metric
    reproduction on the authors' pre-generated audio, we don't need them - but
    we'll need them later for our own ablation runs.
    """
    import zipfile
    from pathlib import Path

    import gdown

    out_dir = Path(VOL_MOUNT) / "lipread_text" / "lrs2_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = out_dir / "lipread_lrs2_test.zip"
    if not (out_dir / "EXTRACTED").exists():
        if not archive.exists() or archive.stat().st_size == 0:
            url = "https://drive.google.com/uc?id=1T4ZFGvW9643844BaOnJyo78-KN83h8Yf"
            print(f"[download] {url}  ->  {archive}")
            gdown.download(url, str(archive), quiet=False)

        # Some gdrive distributions ship plain text in a zip; others ship a flat dir.
        try:
            with zipfile.ZipFile(archive) as z:
                z.extractall(out_dir)
        except zipfile.BadZipFile:
            print(f"[note] {archive} is not a zip - treating as a single .txt blob.")
        (out_dir / "EXTRACTED").touch()
    else:
        print(f"[skip] lipread predictions already extracted under {out_dir}")

    vol.commit()
    print("download_lipread_predictions: done")


# --------------------------------------------------------------------------
# Step 4: Download mahmoodanaam/lrs2 test split + sanity check
# --------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    secrets=[hf_secret],
    timeout=60 * 60,
    cpu=4.0,
    memory=8192,
)
def download_lrs2_test_split():
    """Pull only the `test` split of mahmoodanaam/lrs2 to /vol/lrs2_hf/.

    The full parquet for `test` is ~211 MB (1,243 clips). We pull just this
    split for baseline eval - pretrain/train/val are only needed when we
    start training our modifications.
    """
    from huggingface_hub import snapshot_download

    target = "/vol/lrs2_hf"
    print(f"[download] mahmoodanaam/lrs2 (test only) -> {target}")
    snapshot_download(
        repo_id="mahmoodanaam/lrs2",
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["test/*", "*.json", "README*"],
    )

    vol.commit()
    print("download_lrs2_test_split: done")


@app.function(
    volumes={VOL_MOUNT: vol},
    secrets=[hf_secret],
    timeout=10 * 60,
    cpu=2.0,
    memory=4096,
)
def dataset_sanity_check():
    """Load row 0 of mahmoodanaam/lrs2 test, write the mp4 bytes to disk, ffprobe.

    Confirms (a) the dataset is what we think it is, (b) the binary `video`
    field really is mp4 with embedded audio, (c) fps and resolution are sane.
    """
    import json
    import shutil
    import subprocess
    from pathlib import Path

    from datasets import load_dataset

    ds = load_dataset("/vol/lrs2_hf", split="test", streaming=False)
    print(f"loaded test split: {len(ds)} rows")
    print(f"features: {ds.features}")

    row = ds[0]
    print(f"row[0] keys: {list(row.keys())}")
    print(f"row[0]['sample_id']: {row.get('sample_id')!r}")
    print(f"row[0]['label']: {row.get('label')!r}")
    print(f"row[0]['length']: {row.get('length')!r}")

    video_field = row["video"]
    if isinstance(video_field, dict) and "bytes" in video_field:
        video_bytes = video_field["bytes"]
    elif isinstance(video_field, bytes):
        video_bytes = video_field
    else:
        raise RuntimeError(f"unexpected video field type: {type(video_field)} -> {video_field!r}")

    print(f"video bytes length: {len(video_bytes)}")
    tmp_mp4 = Path("/tmp/lrs2_sample.mp4")
    tmp_mp4.write_bytes(video_bytes)

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe not found in image - bug in image build")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(tmp_mp4)],
        capture_output=True,
        check=True,
        text=True,
    )
    info = json.loads(out.stdout)
    print("ffprobe streams:")
    for s in info.get("streams", []):
        keys = ["codec_type", "codec_name", "width", "height", "r_frame_rate", "sample_rate", "channels"]
        print("  " + ", ".join(f"{k}={s.get(k)}" for k in keys if k in s))

    # Verdict
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_video:
        raise AssertionError("no video stream - dataset format is wrong, pivot to fallback")
    if not has_audio:
        raise AssertionError("no audio stream - dataset format is wrong, pivot to fallback")

    vstream = next(s for s in streams if s["codec_type"] == "video")
    fps_str = vstream.get("r_frame_rate", "0/1")
    num, den = fps_str.split("/")
    fps = int(num) / int(den) if int(den) else 0.0
    print(f"fps: {fps}")
    if abs(fps - 25.0) > 0.5:
        print(f"WARNING: fps != 25 (got {fps}). LipVoicer's pipeline auto-resamples to 25.")
    print("dataset_sanity_check: OK")


# --------------------------------------------------------------------------
# Step 5: Extract LRS2 test videos from parquet to /vol/lrs2_raw/main/
# --------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    secrets=[hf_secret],
    timeout=60 * 60,
    cpu=4.0,
    memory=8192,
)
def extract_test_videos():
    """Write every test clip as <sample_id>.mp4 + <sample_id>.txt under /vol/lrs2_raw/.

    Layout matches what LipVoicer's LipVoicerDataset expects for LRS2:

        /vol/lrs2_raw/main/<video_id>.mp4
        /vol/lrs2_raw/main/<video_id>.txt    ("Text:  <transcript>\n")
        /vol/lrs2_raw/test.txt               (list of video IDs)

    `sample_id` in the HF dataset is expected to look like "<dir>/<clip>"; we
    treat anything before the last "/" as the parent dir, and write under
    `main/`. If the convention turns out to differ, this function will surface
    the actual key format on the first row before it writes anything bulk.
    """
    from pathlib import Path

    from datasets import load_dataset
    from tqdm import tqdm

    ds = load_dataset("/vol/lrs2_hf", split="test", streaming=False)
    print(f"loaded {len(ds)} test rows")

    raw_root = Path(VOL_MOUNT) / "lrs2_raw"
    main_root = raw_root / "main"
    main_root.mkdir(parents=True, exist_ok=True)

    # Print first 3 sample_ids so the format is visible before we commit to a layout.
    for i in range(min(3, len(ds))):
        label_preview = repr(ds[i].get("label"))[:60]
        print(f"sample {i}: id={ds[i].get('sample_id')!r}, label={label_preview}")

    test_ids: list[str] = []
    n_written = 0
    n_skipped = 0
    for row in tqdm(ds, total=len(ds)):
        sid = row["sample_id"]
        # Normalise: strip any leading "main/" or "test/" that may already be present.
        clean = sid.lstrip("/")
        for prefix in ("main/", "test/"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        out_mp4 = main_root / f"{clean}.mp4"
        out_txt = main_root / f"{clean}.txt"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        if out_mp4.exists() and out_mp4.stat().st_size > 0 and out_txt.exists():
            n_skipped += 1
            test_ids.append(clean)
            continue

        video_field = row["video"]
        if isinstance(video_field, dict) and "bytes" in video_field:
            video_bytes = video_field["bytes"]
        elif isinstance(video_field, bytes):
            video_bytes = video_field
        else:
            raise RuntimeError(f"unexpected video field type for {sid}: {type(video_field)}")

        out_mp4.write_bytes(video_bytes)

        transcript = (row.get("label") or "").strip()
        # LRS2 file convention: "Text:  <SENTENCE>\n"
        out_txt.write_text(f"Text:  {transcript}\n")

        test_ids.append(clean)
        n_written += 1

    test_list = raw_root / "test.txt"
    test_list.write_text("\n".join(test_ids) + "\n")
    print(f"wrote {n_written} new clips, skipped {n_skipped}, total {len(test_ids)} in test.txt")

    vol.commit()
    print("extract_test_videos: done")


# --------------------------------------------------------------------------
# Step 6: Sample a deterministic 200-utt eval split
# --------------------------------------------------------------------------
@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=5 * 60,
    cpu=1.0,
    memory=1024,
)
def sample_eval_split(n: int = 200, seed: int = 1234):
    """Pick a deterministic subset of `n` test utterances for the headline eval."""
    import random
    from pathlib import Path

    raw_root = Path(VOL_MOUNT) / "lrs2_raw"
    all_ids = (raw_root / "test.txt").read_text().splitlines()
    all_ids = [x for x in all_ids if x.strip()]
    print(f"total test ids: {len(all_ids)}")

    rng = random.Random(seed)
    chosen = sorted(rng.sample(all_ids, k=min(n, len(all_ids))))

    out = raw_root / f"eval_{n}utt.txt"
    out.write_text("\n".join(chosen) + "\n")
    print(f"wrote {len(chosen)} ids -> {out}")
    vol.commit()


# --------------------------------------------------------------------------
# Step 7: Run the eval harness on the baseline (LipVoicer's pre-generated audio)
# --------------------------------------------------------------------------
@app.function(
    image=GPU_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
)
def eval_baseline(eval_split: str = "eval_200utt.txt"):
    """Compute WER / LSE-C / LSE-D / STOI on LipVoicer's pre-generated LRS2 audio.

    Inputs (all from prior steps):
      - /vol/baseline_audio/lrs2/      (the authors' .wav outputs)
      - /vol/lrs2_raw/main/            (test .mp4 + .txt from extract_test_videos)
      - /vol/lrs2_raw/<eval_split>     (200 selected ids from sample_eval_split)

    Output:
      - /vol/results/baseline_<eval_split>.json
    """
    from experiments.scripts import eval_baseline as runner

    runner.run(
        eval_split=f"{VOL_MOUNT}/lrs2_raw/{eval_split}",
        baseline_audio_root=f"{VOL_MOUNT}/baseline_audio/lrs2",
        videos_root=f"{VOL_MOUNT}/lrs2_raw/main",
        out_path=f"{VOL_MOUNT}/results/baseline_{eval_split.replace('.txt', '')}.json",
    )
    vol.commit()


# ===========================================================================
# TED self-collected pipeline (Phase 1)
# ===========================================================================
# Volume layout established by these functions:
#   /vol/ted_talk_list.txt            URLs to download (~80 talks)
#   /vol/ted_dl/<talk_id>/talk.mp4    Raw downloads (delete after segmentation)
#   /vol/ted_dl/<talk_id>/talk.en.vtt
#   /vol/ted_dl/download_summary.txt
#   /vol/ted_lrs2/raw/main/<talk_id>/<sent_id>.mp4   <- "lrs2" in path so dataset
#   /vol/ted_lrs2/raw/main/<talk_id>/<sent_id>.txt      class auto-detects LRS2 mode
#   /vol/ted_lrs2/raw/{train,val,test,eval_200utt}.txt
#   /vol/ted_lrs2/proc/mouth_roi/main/<talk_id>/<sent_id>.npz
#   /vol/ted_lrs2/proc/audio/main/<talk_id>/<sent_id>.wav
#   /vol/ted_lrs2/proc/audio/main/<talk_id>/<sent_id>.wav.spec
#   /vol/ted_lrs2/proc/lipread_text/<talk_id>/<sent_id>.txt
#   /vol/baseline_audio/ted/<talk_id>/<sent_id>.wav
#   /vol/results/baseline_ted_eval_200utt.json

TED_DL_ROOT = f"{VOL_MOUNT}/ted_dl"
TED_RAW_ROOT = f"{VOL_MOUNT}/ted_lrs2/raw"  # "lrs2" in path -> LRS2 detection in LipVoicerDataset
TED_AUDIO_ROOT = f"{VOL_MOUNT}/ted_lrs2/proc/audio"
TED_MOUTHROI_ROOT = f"{VOL_MOUNT}/ted_lrs2/proc/mouth_roi"
TED_LIPREAD_ROOT = f"{VOL_MOUNT}/ted_lrs2/proc/lipread_text"
TED_BASELINE_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted"
TED_URL_LIST = f"{VOL_MOUNT}/ted_talk_list.txt"

# Flow-only ablation (Run C) paths
TED_FLOW_ROOT = f"{VOL_MOUNT}/ted_lrs2/proc/flow"            # cached RAFT optical flow
TED_FLOW_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_flow"  # flow-tuned generated audio
FLOW_CKPT_ROOT = f"{VOL_MOUNT}/checkpoints/lipvoicer_flow/exp/TED/wnet_h512_d12_T400_betaT0.02/checkpoint"
MELGEN_CKPT = f"{VOL_MOUNT}/checkpoints/lipvoicer/exp/LRS2/wnet_h512_d12_T400_betaT0.02/checkpoint/1000000.pkl"

# Full method (Run D = flow + temporal attention) paths
TED_FULL_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_full"
FULL_CKPT_ROOT = f"{VOL_MOUNT}/checkpoints/lipvoicer_full/exp/TED/wnet_h512_d12_T400_betaT0.02/checkpoint"

# Attention-only (Run B) paths
TED_ATTN_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_attn"
ATTN_CKPT_ROOT = f"{VOL_MOUNT}/checkpoints/lipvoicer_attn/exp/TED/wnet_h512_d12_T400_betaT0.02/checkpoint"

# Speaker/Style consistency (Runs E + F) paths
TED_SPK_REF_CLIPS_JSON = f"{VOL_MOUNT}/ted_lrs2/raw/reference_clips.json"
TED_SPK_EMBED_ROOT = f"{VOL_MOUNT}/ted_lrs2/proc/speaker_embeddings"
TED_SPK_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_spk"
TED_SPK_ATTN_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_spk_attn"
SPK_CKPT_ROOT = f"{VOL_MOUNT}/checkpoints/lipvoicer_spk/exp/TED/wnet_h512_d12_T400_betaT0.02/checkpoint"
SPK_ATTN_CKPT_ROOT = f"{VOL_MOUNT}/checkpoints/lipvoicer_spk_attn/exp/TED/wnet_h512_d12_T400_betaT0.02/checkpoint"
TED_SWAP_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_swap"
TED_SWAP_META_ROOT = f"{VOL_MOUNT}/results/swap_meta"

# FiLM speaker/style + multi-layer denoiser injection (Runs G + H) paths
TED_SPK_FILM_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_spk_film"
TED_SPK_ATTN_FILM_AUDIO_ROOT = f"{VOL_MOUNT}/baseline_audio/ted_spk_attn_film"
SPK_FILM_CKPT_ROOT = f"{VOL_MOUNT}/checkpoints/lipvoicer_spk_film/exp/TED/wnet_h512_d12_T400_betaT0.02/checkpoint"
SPK_ATTN_FILM_CKPT_ROOT = f"{VOL_MOUNT}/checkpoints/lipvoicer_spk_attn_film/exp/TED/wnet_h512_d12_T400_betaT0.02/checkpoint"

# Canonical variant keys used by every train/infer/eval/compare function below:
#   a = baseline (pretrained, no fine-tuning)   b = temporal attention (Run B)
#   c = optical flow (Run C)                    d = flow + attention (Run D)
#   spk = speaker/style ReZero (Run E)          spk_attn = E + attention (Run F)
#   g = FiLM speaker/style (Run G)              h = G + attention (Run H)
AUDIO_ROOTS = {
    "a": TED_BASELINE_AUDIO_ROOT,
    "b": TED_ATTN_AUDIO_ROOT,
    "c": TED_FLOW_AUDIO_ROOT,
    "d": TED_FULL_AUDIO_ROOT,
    "spk": TED_SPK_AUDIO_ROOT,
    "spk_attn": TED_SPK_ATTN_AUDIO_ROOT,
    "g": TED_SPK_FILM_AUDIO_ROOT,
    "h": TED_SPK_ATTN_FILM_AUDIO_ROOT,
}
CKPT_ROOTS = {
    "b": ATTN_CKPT_ROOT,
    "c": FLOW_CKPT_ROOT,
    "d": FULL_CKPT_ROOT,
    "spk": SPK_CKPT_ROOT,
    "spk_attn": SPK_ATTN_CKPT_ROOT,
    "g": SPK_FILM_CKPT_ROOT,
    "h": SPK_ATTN_FILM_CKPT_ROOT,
}
RESULT_PREFIX = {
    "a": "baseline", "b": "attn", "c": "flow", "d": "full",
    "spk": "spk", "spk_attn": "spk_attn", "g": "g", "h": "h",
}
# train_ablation.py / run_ablation_inference.py use descriptive run names.
ABLATION_RUN_NAME = {"b": "attn", "c": "flow", "d": "full"}


def _latest_ckpt(root: str) -> str | None:
    ckpts = sorted(Path(root).glob("*.pkl"), key=lambda p: int(p.stem))
    return str(ckpts[-1]) if ckpts else None


@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=2 * 60 * 60,
    cpu=4.0,
    memory=4096,
)
def fetch_ted_list(n: int = 80, seed: int = 1234, max_candidates: int = 1200):
    """Pull TED sitemap, write ~N talk URLs to /vol/ted_talk_list.txt."""
    from experiments.scripts.fetch_ted_list import run as runner
    runner(out_path=TED_URL_LIST, n=n, seed=seed, max_candidates=max_candidates)
    vol.commit()


@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
    cpu=4.0,
    memory=4096,
)
def download_ted(max_parallel: int = 2, sleep_interval: int = 2):
    """Run yt-dlp over the talk URL list. Idempotent (skips cached)."""
    from experiments.scripts.download_ted import run as runner
    runner(
        url_list_path=TED_URL_LIST,
        dl_root=TED_DL_ROOT,
        max_parallel=max_parallel,
        sleep_interval=sleep_interval,
    )
    vol.commit()


@app.function(
    image=SEGMENT_IMAGE,
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
    cpu=4.0,
    memory=8192,
)
def segment_ted():
    """Use TED GraphQL transcripts + single-face filter + ffmpeg-cut sentences -> /vol/ted_lrs2/raw/main/."""
    from experiments.scripts.segment_ted import run as runner
    runner(
        dl_root=TED_DL_ROOT,
        out_root=TED_RAW_ROOT,
        summary_path=f"{TED_RAW_ROOT}/segment_summary.txt",
    )
    vol.commit()


@app.function(
    image=SEGMENT_IMAGE,
    volumes={VOL_MOUNT: vol},
    timeout=30 * 60,
    cpu=2.0,
    memory=4096,
)
def segment_ted_one(talk_id: str) -> dict:
    """Segment a single TED talk. Idempotent (per-talk _DONE marker), so this
    is safe to fan out with .map() across many CPU containers - each talk
    writes to its own /vol/ted_lrs2/raw/main/<talk_id>/ subtree, so there are
    no cross-container write conflicts on the shared volume.
    """
    from pathlib import Path
    from experiments.scripts.segment_ted import segment_one_talk
    vol.reload()
    stats = segment_one_talk(talk_id, Path(TED_DL_ROOT), Path(TED_RAW_ROOT))
    vol.commit()
    return stats


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=12 * 60 * 60,
)
def process_ted_mouth_rois():
    """Run LipVoicer's face-detection + mouth-ROI + lip-reader on each clip.

    Output:
      /vol/ted_lrs2/proc/mouth_roi/main/<talk>/<sent>.npz   (88x88 grayscale mouth crop)
      /vol/ted_lrs2/proc/lipread_text/<talk>/<sent>.txt     (lip-reader prediction)
    """
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.process_ted_mouth_rois import run as runner
    runner(
        raw_root=TED_RAW_ROOT,
        mouthroi_root=TED_MOUTHROI_ROOT,
        lipread_root=TED_LIPREAD_ROOT,
    )
    vol.commit()


@app.function(
    image=LIPVOICER_IMAGE,
    volumes={VOL_MOUNT: vol},
    timeout=2 * 60 * 60,
    cpu=4.0,
    memory=8192,
)
def extract_ted_mels():
    """Extract 16 kHz mono WAVs + 80-bin mel specs from each TED clip."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.extract_ted_audio_and_mels import run as runner
    runner(raw_root=TED_RAW_ROOT, audio_root=TED_AUDIO_ROOT)
    vol.commit()


@app.function(
    volumes={VOL_MOUNT: vol},
    timeout=10 * 60,
    cpu=2.0,
    memory=2048,
)
def build_ted_manifest(test_talks: int = 10, val_talks: int = 5, eval_n: int = 200):
    """Speaker-disjoint train/val/test split + deterministic eval subset."""
    from experiments.scripts.build_ted_manifest import run as runner
    runner(
        raw_root=TED_RAW_ROOT,
        mouthroi_root=TED_MOUTHROI_ROOT,
        audio_root=TED_AUDIO_ROOT,
        test_talks=test_talks,
        val_talks=val_talks,
        eval_n=eval_n,
    )
    vol.commit()


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
)
def run_ted_baseline_inference(eval_split: str = "eval_200utt.txt", max_clips: int = 0):
    """Run LipVoicer's pretrained MelGen on the TED eval slice -> /vol/baseline_audio/ted/."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.run_baseline_inference import run as runner

    ckpt_root = f"{VOL_MOUNT}/checkpoints/lipvoicer"
    runner(
        eval_split_path=f"{TED_RAW_ROOT}/{eval_split}",
        melgen_ckpt=f"{ckpt_root}/exp/LRS2/wnet_h512_d12_T400_betaT0.02/checkpoint/1000000.pkl",
        videos_dir=TED_RAW_ROOT,
        mouthrois_dir=TED_MOUTHROI_ROOT,
        audios_dir=TED_AUDIO_ROOT,
        lipread_text_dir=TED_LIPREAD_ROOT,
        out_audio_root=TED_BASELINE_AUDIO_ROOT,
        # config.json ships in the LipVoicer repo itself; we run with cwd=/root/repo.
        hifi_gan_config="hifi_gan/config.json",
        hifi_gan_ckpt=f"{ckpt_root}/hifi_gan/g_02400000",
        max_clips=max_clips if max_clips > 0 else None,
    )
    vol.commit()


@app.function(
    image=GPU_IMAGE,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=2 * 60 * 60,
)
def retranscribe_ted():
    """Whisper-transcribe each clip's audio and overwrite the .txt transcript.

    Fixes a misalignment between TED's GraphQL cue timestamps and the actual
    MP4 timeline - the clip audio is ~3 sec offset from the cue text. The
    audio + mouth ROIs are internally consistent (same time window), so
    relabeling clips via Whisper on the audio gives a correct ground truth.
    """
    from experiments.scripts.retranscribe_ted import run as runner
    runner(audio_root=TED_AUDIO_ROOT, raw_root=TED_RAW_ROOT)
    vol.commit()


@app.function(
    image=LIGHT_IMAGE,
    volumes={VOL_MOUNT: vol},
    timeout=16 * 60 * 60,
    cpu=2.0,
    memory=4096,
)
def orchestrate_remainder():
    """Server-side coordinator for the 4 remaining pipeline stages.

    Robustness over the old serial monolith:
      - Stage 1 (segment) fans out across many CPU containers via
        segment_ted_one.map() - turns a 12+ hr serial slog into minutes and
        removes the single-container failure point.
      - Stages 2-4 run as their own .remote() calls (process_ted_mouth_rois on
        its A100, extract_ted_mels + build_ted_manifest on CPU). Each stage and
        each per-clip unit is idempotent, so re-running this function after any
        crash resumes exactly where it stopped - nothing is recomputed.

    This function itself is a cheap CPU coordinator; the GPU work happens inside
    process_ted_mouth_rois's own container, not here.
    """
    from pathlib import Path

    print("=== orchestrate_remainder start ===", flush=True)
    vol.reload()

    # --- talk id list (mirror segment_ted.run's source-of-truth) ---
    dl_path = Path(TED_DL_ROOT)
    summary_file = dl_path / "download_summary.txt"
    if summary_file.exists():
        talk_ids = [t.strip() for t in summary_file.read_text().splitlines() if t.strip()]
    else:
        talk_ids = sorted(p.name for p in dl_path.iterdir() if p.is_dir())
    print(f"talks to consider: {len(talk_ids)}", flush=True)

    # --- Stage 1/4: parallel segmentation ---
    print("\n=== [1/4] segment_ted (parallel .map) ===", flush=True)
    n_done = 0
    n_skipped = 0
    total_kept = 0
    total_considered = 0
    total_seconds = 0.0
    for stats in segment_ted_one.map(talk_ids, return_exceptions=True):
        if isinstance(stats, Exception):
            print(f"  segment worker error: {type(stats).__name__}: {stats}", flush=True)
            continue
        if stats.get("skipped"):
            n_skipped += 1
        else:
            n_done += 1
        total_kept += stats.get("kept", 0)
        total_considered += stats.get("considered", 0)
        total_seconds += stats.get("kept_seconds", 0.0)

    retention = total_kept / max(total_considered, 1)
    print(
        f"segment done: {n_done} processed, {n_skipped} skipped | "
        f"clips {total_kept}/{total_considered} ({retention:.1%}) | "
        f"{total_seconds / 3600:.2f} hr",
        flush=True,
    )
    vol.reload()
    Path(f"{TED_RAW_ROOT}/segment_summary.txt").write_text(
        f"talks: {len(talk_ids)}\n"
        f"clips: {total_kept}\n"
        f"hours: {total_seconds / 3600:.2f}\n"
        f"retention: {retention:.3f}\n"
    )
    vol.commit()

    # --- Stage 2/4: mouth ROIs (own A100 container, resumable) ---
    print("\n=== [2/4] process_ted_mouth_rois (.remote A100) ===", flush=True)
    process_ted_mouth_rois.remote()

    # --- Stage 3/5: mels + 16 kHz audio (own CPU container, resumable) ---
    print("\n=== [3/5] extract_ted_mels (.remote) ===", flush=True)
    extract_ted_mels.remote()

    # --- Stage 4/5: retranscribe (fix TED cue-timestamp ~3s offset) ---
    # segment_ted writes the GraphQL cue text verbatim, but TED's cue timestamps
    # are offset from the MP4 timeline, so the clip audio doesn't match the cue
    # text. Re-Whispering each clip's audio (now extracted in stage 3) gives a
    # ground-truth transcript that matches what's actually spoken in the clip.
    print("\n=== [4/5] retranscribe_ted (.remote A10G) ===", flush=True)
    retranscribe_ted.remote()

    # --- Stage 5/5: manifest ---
    print("\n=== [5/5] build_ted_manifest (.remote) ===", flush=True)
    build_ted_manifest.remote()

    print("\n=== orchestrate_remainder done ===", flush=True)


@app.function(
    image=GPU_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
)
def eval_ted(variant: str = "a", eval_split: str = "eval_200utt.txt"):
    """Score any variant's generated TED audio (WER/STOI/DNSMOS harness)."""
    from experiments.scripts import eval_baseline as runner
    out_name = f"{RESULT_PREFIX[variant]}_ted_" + eval_split.replace(".txt", "") + ".json"
    runner.run(
        eval_split=f"{TED_RAW_ROOT}/{eval_split}",
        baseline_audio_root=AUDIO_ROOTS[variant],
        videos_root=f"{TED_RAW_ROOT}/main",
        out_path=f"{VOL_MOUNT}/results/{out_name}",
    )
    vol.commit()


@app.function(
    image=LIGHT_IMAGE,
    volumes={VOL_MOUNT: vol},
    timeout=10 * 60,
)
def compare_run(variant: str = "c", eval_split: str = "eval_200utt.txt"):
    """Print a variant's metric deltas vs the baseline (Run A)."""
    from experiments.scripts.compare_results import run as runner
    tag = eval_split.replace(".txt", "")
    runner(
        baseline_json=f"{VOL_MOUNT}/results/baseline_ted_{tag}.json",
        flow_json=f"{VOL_MOUNT}/results/{RESULT_PREFIX[variant]}_ted_{tag}.json",
    )


# --------------------------------------------------------------------------
# Flow-only ablation (Run C): RAFT optical-flow stream, fine-tuned vs frozen baseline
# --------------------------------------------------------------------------
@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
)
def precompute_flow_one(talk_id: str) -> dict:
    """RAFT optical flow for one talk's clips. Idempotent -> safe for .map() fan-out."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.precompute_flow import run_talk
    vol.reload()
    stats = run_talk(talk_id, TED_MOUTHROI_ROOT, TED_FLOW_ROOT)
    vol.commit()
    return stats


@app.function(
    image=LIGHT_IMAGE,
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
    cpu=2.0,
    memory=4096,
)
def precompute_flow():
    """Coordinator: fan out RAFT flow precompute across talks (parallel, resumable)."""
    from pathlib import Path
    vol.reload()
    talk_ids = sorted(p.name for p in (Path(TED_MOUTHROI_ROOT) / "main").iterdir() if p.is_dir())
    print(f"precompute flow over {len(talk_ids)} talks", flush=True)
    ok = fail = 0
    for s in precompute_flow_one.map(talk_ids, return_exceptions=True):
        if isinstance(s, Exception):
            print(f"  worker error: {type(s).__name__}: {s}", flush=True)
            continue
        ok += s.get("ok", 0)
        fail += s.get("fail", 0)
    print(f"flow precompute done: {ok} clips ok, {fail} failed", flush=True)
    vol.commit()


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=30 * 60,
)
def check_invariance(variant: str = "c"):
    """Step-0 invariance gate: the variant's new modules must compose to identity
    at init. variant in {c, d, spk, spk_attn, g, h}."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.check_invariance import run as runner
    runner(
        variant=ABLATION_RUN_NAME.get(variant, variant),
        melgen_ckpt=MELGEN_CKPT,
        videos_dir=TED_RAW_ROOT,
        mouthrois_dir=TED_MOUTHROI_ROOT,
        audios_dir=TED_AUDIO_ROOT,
        flow_dir=TED_FLOW_ROOT,
        speaker_ref_dir=TED_SPK_EMBED_ROOT,
        ref_clips_json=TED_SPK_REF_CLIPS_JSON,
    )


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=12 * 60 * 60,
)
def train_ablation(variant: str = "c", steps: int = 6000, batch: int = 16, lr: float = 2e-4,
                   lam0: float = 1.0, anneal: int = 2000):
    """Fine-tune Run B (attention), C (flow), or D (flow + attention); everything
    else frozen. D warm-starts flow_fusion from the latest Run C checkpoint."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.train_ablation import run as runner

    flow_ckpt = _latest_ckpt(FLOW_CKPT_ROOT) if variant == "d" else None
    if variant == "d":
        print(f"warm-start from flow ckpt: {flow_ckpt}")
    runner(
        run_name=ABLATION_RUN_NAME[variant],
        melgen_ckpt=MELGEN_CKPT,
        videos_dir=TED_RAW_ROOT,
        mouthrois_dir=TED_MOUTHROI_ROOT,
        audios_dir=TED_AUDIO_ROOT,
        flow_dir=TED_FLOW_ROOT,
        save_dir=CKPT_ROOTS[variant],
        flow_ckpt=flow_ckpt,
        steps=steps, batch=batch, lr=lr, lam0=lam0, anneal=anneal,
    )
    vol.commit()


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
)
def infer_ablation(variant: str = "c", eval_split: str = "eval_200utt.txt", max_clips: int = 0):
    """Generate Run B/C/D audio on the TED eval slice from the latest checkpoint."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.run_ablation_inference import run as runner

    ckpt = _latest_ckpt(CKPT_ROOTS[variant])
    if not ckpt:
        raise SystemExit(f"no {variant} checkpoint found in {CKPT_ROOTS[variant]}")
    print(f"using {variant} ckpt: {ckpt}")
    runner(
        run_name=ABLATION_RUN_NAME[variant],
        eval_split_path=f"{TED_RAW_ROOT}/{eval_split}",
        ckpt=ckpt,
        videos_dir=TED_RAW_ROOT,
        mouthrois_dir=TED_MOUTHROI_ROOT,
        audios_dir=TED_AUDIO_ROOT,
        flow_dir=TED_FLOW_ROOT,
        lipread_text_dir=TED_LIPREAD_ROOT,
        out_audio_root=AUDIO_ROOTS[variant],
        hifi_gan_config="hifi_gan/config.json",
        hifi_gan_ckpt=f"{VOL_MOUNT}/checkpoints/lipvoicer/hifi_gan/g_02400000",
        max_clips=max_clips if max_clips > 0 else None,
    )
    vol.commit()


# --------------------------------------------------------------------------
# Speaker/Style Consistency (Runs E + F)
# --------------------------------------------------------------------------
@app.function(
    image=GPU_IMAGE,  # has soundfile + numpy, no GPU needed but reuses lightweight Python deps
    volumes={VOL_MOUNT: vol},
    timeout=3 * 60 * 60,
    cpu=8.0,
)
def select_reference_clips(top_k: int = 5):
    """Score each clip for speaker-reference quality and write per-talk ranking JSON."""
    from experiments.scripts.select_reference_clips import run as runner
    runner(audio_root=TED_AUDIO_ROOT, raw_root=TED_RAW_ROOT, top_k=top_k)
    vol.commit()


@app.function(
    image=SPEAKERSIM_IMAGE,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
)
def precompute_speaker_embeddings(top_k: int = 5):
    """Cache 192-d ECAPA embeddings for the top-K clean clips per talk."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.precompute_speaker_embeddings import run as runner
    runner(audio_root=TED_AUDIO_ROOT, raw_root=TED_RAW_ROOT, embed_root=TED_SPK_EMBED_ROOT, top_k=top_k)
    vol.commit()


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=12 * 60 * 60,
    retries=modal.Retries(max_retries=10, initial_delay=30.0, backoff_coefficient=1.5),
)
def train_speaker(variant: str = "spk", steps: int = 6000, batch: int = 16, lr: float = 2e-4,
                  lam_contrastive: float = 0.5, temperature: float = 0.07,
                  val_every: int = 200, patience: int = 8,
                  ema_alpha: float = 0.3, ema_eps: float = 1e-3):
    """Fine-tune a speaker/style variant: spk/spk_attn (Runs E/F, ReZero fusion) or
    g/h (Runs G/H, FiLM fusion + per-layer FiLM into the frozen denoiser; these use
    an EMA-based stop criterion over val_loss AND contrastive loss)."""
    import sys
    sys.path.insert(0, "/root/repo")

    save_dir = CKPT_ROOTS[variant]
    # Resume preference: own dir > Run B's attention ckpt (attention variants) > cold start.
    attn_ckpt = _latest_ckpt(save_dir)
    if attn_ckpt:
        print(f"RESUMING {variant} from prior checkpoint: {attn_ckpt}")
    elif variant in ("spk_attn", "h"):
        attn_ckpt = _latest_ckpt(ATTN_CKPT_ROOT)
        print(f"cold start {variant}, warm attention from: {attn_ckpt}")

    common = dict(variant=variant, melgen_ckpt=MELGEN_CKPT, attn_ckpt=attn_ckpt,
                  videos_dir=TED_RAW_ROOT, mouthrois_dir=TED_MOUTHROI_ROOT,
                  audios_dir=TED_AUDIO_ROOT, flow_dir=TED_FLOW_ROOT,
                  speaker_ref_dir=TED_SPK_EMBED_ROOT, ref_clips_json=TED_SPK_REF_CLIPS_JSON,
                  save_dir=save_dir, steps=steps, batch=batch, lr=lr,
                  lam_contrastive=lam_contrastive, temperature=temperature,
                  val_every=val_every, patience=patience)
    if variant in ("g", "h"):
        from experiments.scripts.train_spk_film import run as runner
        runner(ema_alpha=ema_alpha, ema_eps=ema_eps, **common)
    else:
        from experiments.scripts.train_spk import run as runner
        runner(**common)
    vol.commit()


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
)
def infer_speaker(variant: str = "spk", eval_split: str = "eval_200utt.txt", max_clips: int = 0):
    """Generate speaker/style audio (Runs E/F/G/H) on the TED eval split."""
    import sys
    sys.path.insert(0, "/root/repo")

    ckpt = _latest_ckpt(CKPT_ROOTS[variant])
    if not ckpt:
        raise SystemExit(f"no {variant} checkpoint found in {CKPT_ROOTS[variant]}")
    print(f"using {variant} ckpt: {ckpt}")

    common = dict(variant=variant, eval_split_path=f"{TED_RAW_ROOT}/{eval_split}",
                  videos_dir=TED_RAW_ROOT, mouthrois_dir=TED_MOUTHROI_ROOT,
                  audios_dir=TED_AUDIO_ROOT, flow_dir=TED_FLOW_ROOT,
                  speaker_ref_dir=TED_SPK_EMBED_ROOT, ref_clips_json=TED_SPK_REF_CLIPS_JSON,
                  lipread_text_dir=TED_LIPREAD_ROOT, out_audio_root=AUDIO_ROOTS[variant],
                  hifi_gan_config="hifi_gan/config.json",
                  hifi_gan_ckpt=f"{VOL_MOUNT}/checkpoints/lipvoicer/hifi_gan/g_02400000",
                  max_clips=max_clips if max_clips > 0 else None)
    if variant in ("g", "h"):
        from experiments.scripts.run_spk_film_inference import run as runner
        runner(ckpt=ckpt, **common)
    else:
        from experiments.scripts.run_spk_inference import run as runner
        runner(spk_ckpt=ckpt, **common)
    vol.commit()


@app.function(
    image=SPEAKERSIM_IMAGE,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
)
def compute_gen_embeddings(variant: str, eval_split: str = "eval_200utt.txt", out_name: str = ""):
    """Compute ECAPA embeddings for every generated wav of a variant and save as a single
    .npy + metadata json on /vol/results/. Used for the t-SNE figure."""
    import json
    import sys
    from pathlib import Path
    import numpy as np
    import torch
    import torchaudio
    sys.path.insert(0, "/root/repo")
    from speechbrain.inference.speaker import EncoderClassifier

    gen_root = Path(AUDIO_ROOTS[variant])
    eval_ids = [x.strip().removeprefix("main/") for x in Path(f"{TED_RAW_ROOT}/{eval_split}").read_text().splitlines() if x.strip()]

    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda:0"},
    )
    ecapa.eval()

    embs, vids, talks = [], [], []
    with torch.no_grad():
        for vid in eval_ids:
            wav = gen_root / f"{vid}.wav"
            if not wav.exists():
                continue
            audio, sr = torchaudio.load(str(wav))
            if audio.dim() == 2 and audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            if audio.size(-1) > 16000 * 10:
                audio = audio[..., :16000 * 10]
            audio = audio.cuda()
            e = ecapa.encode_batch(audio).squeeze().detach().cpu().numpy()
            embs.append(e)
            vids.append(vid)
            talks.append(vid.split("/")[0])

    arr = np.stack(embs, axis=0)
    tag = out_name or f"gen_emb_{variant}"
    out_npy = Path(f"{VOL_MOUNT}/results/{tag}.npy")
    out_json = out_npy.with_suffix(".json")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), arr)
    out_json.write_text(json.dumps({"variant": variant, "video_ids": vids, "talks": talks, "n": len(vids)}, indent=2))
    print(f"saved {out_npy} | shape {arr.shape} | n={len(vids)} | unique talks={len(set(talks))}")
    vol.commit()


@app.function(
    image=SPEAKERSIM_IMAGE,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
)
def eval_speaker_sim(variant: str = "spk", eval_split: str = "eval_200utt.txt"):
    """ECAPA cosine-similarity eval for any generated audio (variant in {a, b, c, d, spk, spk_attn, g, h})."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.eval_speaker_similarity import run as runner
    gen_root = AUDIO_ROOTS[variant]
    tag = eval_split.replace(".txt", "")
    runner(eval_split_path=f"{TED_RAW_ROOT}/{eval_split}", gen_audio_root=gen_root,
           ref_clips_json=TED_SPK_REF_CLIPS_JSON, speaker_emb_root=TED_SPK_EMBED_ROOT,
           out_path=f"{VOL_MOUNT}/results/spk_sim_{variant}_{tag}.json")
    vol.commit()


@app.function(
    image=LIPVOICER_IMAGE,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: vol},
    timeout=4 * 60 * 60,
)
def swap_test_generate(variant: str = "spk", eval_split: str = "eval_200utt.txt", n_pairs: int = 50):
    """PHASE 1: cross-speaker generation - A's video + B's audio reference -> swap.wav,
    plus A's video + A's audio reference -> nonswap.wav. Writes a metadata JSON listing
    each generated pair for scoring in PHASE 2. No speechbrain in this image.
    variant in {spk, spk_attn, g, h}."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.swap_test import run as runner

    spk_ckpt = _latest_ckpt(CKPT_ROOTS[variant])
    if not spk_ckpt:
        raise SystemExit(f"no {variant} checkpoint found in {CKPT_ROOTS[variant]}")
    out_audio = f"{TED_SWAP_AUDIO_ROOT}/{variant}"
    meta_json = f"{TED_SWAP_META_ROOT}/swap_meta_{variant}.json"
    Path(TED_SWAP_META_ROOT).mkdir(parents=True, exist_ok=True)
    runner(variant=variant, eval_split_path=f"{TED_RAW_ROOT}/{eval_split}",
           spk_ckpt=spk_ckpt, videos_dir=TED_RAW_ROOT, mouthrois_dir=TED_MOUTHROI_ROOT,
           audios_dir=TED_AUDIO_ROOT, flow_dir=TED_FLOW_ROOT,
           speaker_ref_dir=TED_SPK_EMBED_ROOT, ref_clips_json=TED_SPK_REF_CLIPS_JSON,
           lipread_text_dir=TED_LIPREAD_ROOT, out_audio_root=out_audio,
           out_json_path=meta_json,
           hifi_gan_config="hifi_gan/config.json",
           hifi_gan_ckpt=f"{VOL_MOUNT}/checkpoints/lipvoicer/hifi_gan/g_02400000",
           n_pairs=n_pairs)
    vol.commit()


@app.function(
    image=SPEAKERSIM_IMAGE,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=60 * 60,
)
def swap_test_score(variant: str = "spk"):
    """PHASE 2: ECAPA-score each generated swap.wav + nonswap.wav vs the precomputed
    speaker embeddings for talks A and B. Produces the final swap_test_<variant>.json.
    variant in {spk, spk_attn, g, h}."""
    import sys
    sys.path.insert(0, "/root/repo")
    from experiments.scripts.swap_test_score import run as runner

    meta_json = f"{TED_SWAP_META_ROOT}/swap_meta_{variant}.json"
    out_json = f"{VOL_MOUNT}/results/swap_test_{variant}.json"
    runner(meta_json_path=meta_json, speaker_emb_root=TED_SPK_EMBED_ROOT,
           out_json_path=out_json)
    vol.commit()


@app.function(
    image=GPU_IMAGE,
    volumes={VOL_MOUNT: vol},
    timeout=20 * 60,
    cpu=4.0,
)
def dataset_hours():
    """Sum actual clip durations per manifest split (reads .wav headers only)."""
    import soundfile as sf
    from pathlib import Path
    audio_main = Path(TED_AUDIO_ROOT) / "main"
    grand_n, grand_s = 0, 0.0
    for split in ["train", "val", "test", "eval_200utt"]:
        f = Path(TED_RAW_ROOT) / f"{split}.txt"
        if not f.exists():
            continue
        ids = [x.strip().removeprefix("main/") for x in f.read_text().splitlines() if x.strip()]
        total, n = 0.0, 0
        for vid in ids:
            wav = audio_main / f"{vid}.wav"
            if wav.exists():
                info = sf.info(str(wav))
                total += info.frames / info.samplerate
                n += 1
        if split != "eval_200utt":
            grand_n += n
            grand_s += total
        print(f"{split:12s}: {n:5d} clips  {total/3600:6.2f} hr  ({total:7.0f}s)  avg {total/max(n,1):.2f}s")
    print(f"{'TOTAL':12s}: {grand_n:5d} clips  {grand_s/3600:6.2f} hr  (train+val+test)")


@app.function(
    image=GPU_IMAGE,
    gpu="A10G",
    volumes={VOL_MOUNT: vol},
    timeout=2 * 60 * 60,
)
def align_ted_labels(eval_split: str = "eval_200utt.txt"):
    """Re-derive eval-clip labels from TED's transcript (correctly aligned)."""
    from experiments.scripts.align_ted_labels import run as runner
    runner(
        eval_split_path=f"{TED_RAW_ROOT}/{eval_split}",
        audio_root=TED_AUDIO_ROOT,
        raw_root=TED_RAW_ROOT,
        dl_root=TED_DL_ROOT,
    )
    vol.commit()


@app.local_entrypoint()
def main(step: str = "", variant: str = "", eval_split: str = ""):
    """Dispatch any app function by name:

        modal run modal_app.py --step <function> [--variant <v>] [--eval-split <file>]

    e.g.
        modal run modal_app.py --step dataset_sanity_check
        modal run modal_app.py --step train_ablation --variant c
        modal run modal_app.py --step eval_ted --variant h --eval-split eval_clean.txt

    Omit --variant / --eval-split for functions that don't take them (each
    function's own defaults apply). Run with no --step to list all steps.
    """
    fn = globals().get(step)
    if fn is None or not hasattr(fn, "remote"):
        steps = sorted(name for name, obj in globals().items()
                       if hasattr(obj, "remote") and not name.startswith("_"))
        raise SystemExit(
            "usage: modal run modal_app.py --step <name> [--variant v] [--eval-split f]\n"
            "steps:\n  " + "\n  ".join(steps)
        )
    kwargs = {}
    if variant:
        kwargs["variant"] = variant
    if eval_split:
        kwargs["eval_split"] = eval_split
    fn.remote(**kwargs)
