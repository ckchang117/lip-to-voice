# some parts of this code were borrowed from https://github.com/facebookresearch/VisualVoice/tree/main
# under the licence https://github.com/facebookresearch/VisualVoice/blob/main/LICENSE


import json
import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from scipy.io.wavfile import read
from glob import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
from .video_reader import VideoReader
from .lipreading_utils import *
import cv2
import torchaudio
import torchvision.transforms as transforms
from .stft import normalise_mel


def files_to_list(data_path, suffix):
    """
    Load all .wav files in data_path
    """
    files = glob(os.path.join(data_path, f'**/*.{suffix}'), recursive=True)
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class LipVoicerDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, split, videos_dir, mouthrois_dir, audios_dir, sampling_rate, videos_window_size, audio_stft_hop,
                 flow_dir=None, speaker_ref_dir=None, ref_clips_json=None, face_seq_n=8):
        self.mouthrois_dir = mouthrois_dir
        self.flow_dir = flow_dir
        self.speaker_ref_dir = speaker_ref_dir
        self.face_seq_n = face_seq_n
        if ref_clips_json is not None and Path(ref_clips_json).exists():
            self.ref_clips = json.loads(Path(ref_clips_json).read_text())
        else:
            self.ref_clips = None
        if "LRS3" in videos_dir:
            self.ds_name = "LRS3"
            split_dir = ['pretrain','trainval'] if split in ['train', 'val'] else ['test']
            self.videos_dir = videos_dir
            self.audios_dir = audios_dir
            
            self.moutroi_files = []
            for s in split_dir:
                _mouthrois_dir = os.path.join(mouthrois_dir, s)
                self.moutroi_files += files_to_list(_mouthrois_dir, 'npz')

        elif "LRS2" in videos_dir:
            self.ds_name = "LRS2"
            split_dir = ['main']
            if split == 'train':
                split_dir.append('pretrain')
            
            self.moutroi_files = []
            for s in split_dir:
                if s == "main":
                    if split == "pretrain":
                        videos_list_file = os.path.join(videos_dir, "train.txt")
                    else:
                        videos_list_file = os.path.join(videos_dir, split+".txt")
                    with open(videos_list_file, "r") as f:
                        _video_ids = f.readlines()
                    for _vid in _video_ids:
                        mouthroi_file = os.path.join(mouthrois_dir, s, _vid.strip("\n")+".npz")
                        mouthroi_file = mouthroi_file.replace(" NF", "").replace(" MV", "")
                        if os.path.isfile(mouthroi_file):
                            self.moutroi_files.append(mouthroi_file)
                elif s == "pretrain":
                    self.moutroi_files += glob(os.path.join(mouthrois_dir, "pretrain", "**/*.npz"), recursive=True)
                    
            self.videos_dir = videos_dir
            self.audios_dir = audios_dir
            self.moutroi_files = sorted(self.moutroi_files)        
        
        self.test = True if split=='test' else False
        self.videos_window_size = videos_window_size
        self.audio_stft_hop = audio_stft_hop
        random.seed(1234)
        random.shuffle(self.moutroi_files)
        self.sampling_rate = sampling_rate

        # When the flow stream is active, mouthroi and flow must share the SAME
        # spatial crop so their pixels line up. RandomCrop + HorizontalFlip draw
        # independent params per call, so for the flow path we use the
        # deterministic center-crop ('val') transform on mouthroi for every split.
        transform_split = 'val' if (flow_dir is not None and split == 'train') else split
        self.mouthroi_transform = self.get_mouthroi_transform()[transform_split]
        self.face_image_transform = self.get_face_image_transform()
        if flow_dir is not None:
            # TED frames are 16:9, so Resize(224) yields variable-width faces that
            # can't be batch-stacked. Center-crop to a square 224x224 — this also
            # matches the center-crop the inference scripts apply, keeping the frozen
            # face encoder's input distribution identical between train and test.
            self.face_image_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index):
        while True:
            # Get paths
            mouthroi_filename = self.moutroi_files[index]
            pfilename = Path(mouthroi_filename)
            if self.ds_name in ["LRS3", "LRS2"]:
                video_id = '/'.join([pfilename.parts[-2], pfilename.stem])
                video_filename = mouthroi_filename.replace(self.mouthrois_dir, self.videos_dir).replace('.npz','.mp4')
                melspec_filename = mouthroi_filename.replace(self.mouthrois_dir, self.audios_dir).replace('.npz','.wav.spec')
            
            # Get mouthroi
            mouthroi = np.load(mouthroi_filename)['data']
            if mouthroi.shape[0] >= self.videos_window_size or self.test:
                break
            else:
                index = random.randrange(len(self.moutroi_files))
        melspec = torch.load(melspec_filename)
        face_image = self.load_frame(video_filename)
        
        video = cv2.VideoCapture(video_filename)
        info = {'audio_fps': self.sampling_rate, 'video_fps': video.get(cv2.CAP_PROP_FPS)}

        if self.test:
            audio, fs = torchaudio.load(melspec_filename.replace('.spec', ''))
            text_filename = video_filename.replace(".mp4", ".txt")
            text = self.preprocess_text(text_filename)

            # Normalisations & transforms
            audio = audio / 1.1 / audio.abs().max()
            face_image = self.face_image_transform(face_image)
            mouthroi = torch.FloatTensor(self.mouthroi_transform(mouthroi)).unsqueeze(0)
            melspec = normalise_mel(melspec)

            if self.flow_dir is not None:
                flow = self._load_flow(mouthroi_filename)                # [T_full,2,H,W]
                flow = self._center_crop_flow(flow)                      # [T_full,2,88,88]
                flow = torch.tensor(flow, dtype=torch.float32).permute(1, 0, 2, 3).contiguous()  # [2,T_full,88,88]
                if self.speaker_ref_dir is not None and self.ref_clips is not None:
                    talk_id, clip_id = video_id.split('/')
                    face_seq = self._load_face_seq(video_filename)         # (M,3,224,224)
                    ref_emb = torch.from_numpy(self._pick_reference_embedding(talk_id, clip_id))
                    return (melspec, audio, mouthroi, face_image, flow, face_seq, ref_emb, text, video_id)
                return (melspec, audio, mouthroi, face_image, flow, text, video_id)
            return (melspec, audio, mouthroi, face_image, text, video_id)
        else:

            # Get corresponding crops
            mouthroi, melspec, st_fr = self.extract_window(mouthroi, melspec, info)
            if mouthroi.shape[0] < self.videos_window_size:
                return self.__getitem__(random.randrange(len(self)))

            # Augmentations
            face_image = self.augment_image(face_image)

            # Noramlisations & Transforms
            face_image = self.face_image_transform(face_image)
            mouthroi = torch.FloatTensor(self.mouthroi_transform(mouthroi)).unsqueeze(0)   # add channel dim
            melspec = normalise_mel(melspec)

            if self.flow_dir is not None:
                flow = self._load_flow(mouthroi_filename)                     # [T_full,2,H,W]
                flow = flow[st_fr:st_fr + self.videos_window_size]            # [25,2,H,W], same st_fr
                flow = self._center_crop_flow(flow)                           # [25,2,88,88]
                flow = torch.tensor(flow, dtype=torch.float32).permute(1, 0, 2, 3).contiguous()  # [2,25,88,88]
                if self.speaker_ref_dir is not None and self.ref_clips is not None:
                    talk_id, clip_id = video_id.split('/')
                    face_seq = self._load_face_seq(video_filename)             # (M,3,224,224)
                    ref_emb = torch.from_numpy(self._pick_reference_embedding(talk_id, clip_id))
                    return (melspec, mouthroi, face_image, flow, face_seq, ref_emb)
                return (melspec, mouthroi, face_image, flow)
            return (melspec, mouthroi, face_image)

    def __len__(self):
        return len(self.moutroi_files)

    def extract_window(self, mouthroi, mel, info):
        hop = self.audio_stft_hop

        # vid : T,C,H,W
        vid_2_aud = info['audio_fps'] / info['video_fps'] / hop

        st_fr = random.randint(0, mouthroi.shape[0] - self.videos_window_size)
        mouthroi = mouthroi[st_fr:st_fr + self.videos_window_size]

        st_mel_fr = int(st_fr * vid_2_aud)
        mel_window_size = int(self.videos_window_size * vid_2_aud)

        mel = mel[:, st_mel_fr:st_mel_fr + mel_window_size]

        # st_fr returned so the flow stream can window with the IDENTICAL offset.
        return mouthroi, mel, st_fr

    @staticmethod
    def _center_crop_flow(flow, crop=88):
        """Center-crop the spatial dims of a [T, 2, H, W] flow array to crop x crop."""
        _, _, h, w = flow.shape
        top = (h - crop) // 2
        left = (w - crop) // 2
        return flow[:, :, top:top + crop, left:left + crop]

    def _load_flow(self, mouthroi_filename):
        flow_filename = mouthroi_filename.replace(self.mouthrois_dir, self.flow_dir)
        return np.load(flow_filename)['data']  # [T_full, 2, H, W] float16

    def _load_face_seq(self, video_filename, M=None):
        """Extract M face crops at evenly-spaced timestamps; (M, 3, 224, 224) after transform."""
        if M is None:
            M = self.face_seq_n
        cap = cv2.VideoCapture(video_filename)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        out = []
        for i in range(M):
            if total > 0:
                idx = min(int(i * total / M), total - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
            else:
                ok = False
                frame = None
            if not ok:
                # Fallback: repeat the previous frame if available, else zeros.
                if out:
                    out.append(out[-1])
                    continue
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).convert("RGB")
            out.append(self.face_image_transform(img))  # (3,224,224)
        cap.release()
        return torch.stack(out, dim=0)  # (M, 3, 224, 224)

    def _pick_reference_embedding(self, talk_id: str, target_clip_id: str, top_k: int = 5):
        """Sample a reference clip from the talk's top-K clean pool, excluding the target."""
        if self.ref_clips is None or self.speaker_ref_dir is None:
            return np.zeros(192, dtype=np.float32)
        pool = self.ref_clips.get(talk_id, [])[:top_k]
        candidates = [cid for cid, _ in pool if cid != target_clip_id]
        if not candidates:
            # No alternatives available; use the top clip or fall back to zeros.
            candidates = [pool[0][0]] if pool else None
        if candidates is None:
            return np.zeros(192, dtype=np.float32)
        ref_clip_id = random.choice(candidates)
        emb_path = Path(self.speaker_ref_dir) / "main" / talk_id / f"{ref_clip_id}.npy"
        if not emb_path.exists():
            return np.zeros(192, dtype=np.float32)
        return np.load(str(emb_path)).astype(np.float32)

    @staticmethod
    def load_frame(clip_path):
        video_reader = VideoReader(clip_path, 1)
        start_pts, time_base, total_num_frames = video_reader._compute_video_stats()
        end_frame_index = total_num_frames - 1
        if end_frame_index < 0:
            clip = video_reader.read_video_only(start_pts, 1)
        else:
            clip = video_reader.read_video_only(random.randint(0, end_frame_index) * time_base, 1)
        frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')
        return frame

    @staticmethod
    def augment_image(image):
        if(random.random() < 0.5):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        return image

    @staticmethod
    def get_mouthroi_transform():
        # -- preprocess for the video stream
        preprocessing = {}
        # -- LRW config
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])
        preprocessing['val'] = Compose([
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])
        preprocessing['test'] = preprocessing['val']
        return preprocessing
    
    @staticmethod
    def get_face_image_transform():
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.Resize(224), transforms.ToTensor(), normalize]
        vision_transform = transforms.Compose(vision_transform_list)
        return vision_transform
    
    @staticmethod
    def preprocess_text(txt_filename):
        with open(txt_filename, "r") as f:
            txt = f.readline()[7:]  # discard 'Text:  ' prefix
        txt = txt.replace("{LG}", "")  # remove laughter
        txt = txt.replace("{NS}", "")  # remove noise
        txt = txt.replace("\n", "")
        txt = txt.replace("  ", " ")
        txt = txt.lower().strip()
        return txt

