import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import torchvision.transforms as transforms

class AVHDDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.dataset = config.dataset
        self.split = split

        self.audio_emb_dir = config.audio_dir
        self.visual_emb_dir = config.visual_dir
        self.label_dir = config.label_dir
        self.raw_audio_dir = config.raw_audio_dir
        self.raw_video_dir = getattr(config, 'raw_video_dir', None)

        if self.dataset == "tvsum":
            if split == 'train' and hasattr(config, 'train_keys'):
                self.video_ids = config.train_keys
            elif split == 'val' and hasattr(config, 'val_keys'):
                self.video_ids = config.val_keys
            elif split == 'test' and hasattr(config, 'test_keys'):
                self.video_ids = config.test_keys
            else:
                print(f"Warning: Split keys for '{split}' not found in config. Using fallback 80/10/10 split for TVSum.")
                self.video_ids = self.create_tvsum_split(config.seed)

        elif self.dataset == "mrhisum":
            with open(config.split_file, "r") as f:
                splits = json.load(f)
            self.video_ids = splits[f"{split}_keys"]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        # These parameters should match TFD Conv's input requirements and audio feature's temporal resolution
        self.mel_spectrogram_transformer = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.audio_config['sample_rate'],
            n_fft=config.audio_config['n_fft'],
            win_length=config.audio_config['win_length'],
            hop_length=config.audio_config['hop_length'],
            n_mels=config.audio_config['n_mels']
        ).to(torch.device('cpu'))

        self.frame_resize_dim = config.visual_config.get('resize_dim', 224) 
        self.video_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.frame_resize_dim, self.frame_resize_dim)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.video_ids, self.raw_audio_paths, self.raw_video_paths = self._filter_existing_files()
        print(f"[{split.upper()}] Total usable videos: {len(self.video_ids)}")
        
    
    def _find_raw_audio_file(self, vid):
        supported_extensions = ['.m4a', '.webm', '.mp4', '.wav', '.mp3']
        for ext in supported_extensions:
            path = os.path.join(self.raw_audio_dir, vid + ext)
            if os.path.exists(path):
                return path
        return None
    

    def _find_raw_video_file(self, vid):
            supported_extensions = ['.mp4', '.avi', '.mkv', '.webm']
            if self.raw_video_dir:
                for ext in supported_extensions:
                    path = os.path.join(self.raw_video_dir, vid + ext)
                    if os.path.exists(path):
                        return path


                    
            return None
    
    def _filter_existing_files(self):
            valid_ids = []
            valid_raw_audio_paths = []
            valid_raw_video_paths = []

            for vid in self.video_ids:
                audio_emb_path = os.path.join(self.audio_emb_dir, f"{vid}.npy")
                visual_emb_path = os.path.join(self.visual_emb_dir, f"{vid}.npy")
                label_path = os.path.join(self.label_dir, f"{vid}.npy")
                raw_audio_path = self._find_raw_audio_file(vid)
                raw_video_path = self._find_raw_video_file(vid)
                if (os.path.exists(audio_emb_path) and
                    os.path.exists(visual_emb_path) and
                    os.path.exists(label_path) and
                    raw_audio_path is not None and
                    raw_video_path is not None):

                    valid_ids.append(vid)
                    valid_raw_audio_paths.append(raw_audio_path)
                    valid_raw_video_paths.append(raw_video_path)

            return valid_ids, valid_raw_audio_paths, valid_raw_video_paths

    def create_tvsum_split(self, seed):
        """Fallback split for non-k-fold scenarios. Splits into 70/10/20."""
        all_ids = sorted([f[:-4] for f in os.listdir(self.audio_emb_dir) if f.endswith(".npy")])
        
        train_val_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=seed)
        
        # (1/8) of 80% is 10% of total
        train_ids, val_ids = train_test_split(train_val_ids, test_size=(1/8), random_state=seed)

        if self.split == "train":
            return train_ids
        elif self.split == "val":
            return val_ids
        else:
            return test_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]

        audio_feat_path = os.path.join(self.audio_emb_dir, vid + ".npy")
        visual_feat_path = os.path.join(self.visual_emb_dir, vid + ".npy")
        label_path = os.path.join(self.label_dir, vid + ".npy")

        audio_feat = np.load(audio_feat_path)
        visual_feat = np.load(visual_feat_path)
        label = np.load(label_path)

        if audio_feat.ndim == 3:
            audio_feat = audio_feat.squeeze(0)

        T = min(len(audio_feat), len(visual_feat), len(label))
        
        audio_feat = audio_feat[:T]
        visual_feat = visual_feat[:T]
        label = label[:T]

        try:
            raw_audio_path = self.raw_audio_paths[idx]
            waveform, sr = torchaudio.load(raw_audio_path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.config.audio_config['sample_rate']:
                resampler = torchaudio.transforms.Resample(sr, self.config.audio_config['sample_rate'])
                waveform = resampler(waveform)

            mel_spec = self.mel_spectrogram_transformer(waveform)
            mel_spec = mel_spec.squeeze(0)

            mel_spec = (mel_spec + 1e-8).log()

            frames_per_sec = self.config.audio_config['sample_rate'] / self.config.audio_config['hop_length']

            expected_mel_frames = int(T * frames_per_sec)
            current_mel_frames = mel_spec.shape[1]

            if current_mel_frames > expected_mel_frames:
                aligned_mel_spec = mel_spec[:, :expected_mel_frames]
            elif current_mel_frames < expected_mel_frames:
                padding_needed = expected_mel_frames - current_mel_frames
                aligned_mel_spec = F.pad(mel_spec, (0, padding_needed), 'constant', 0)
            else:
                aligned_mel_spec = mel_spec

            target_fps = self.config.audio_config['target_fps']
            downsample_factor = int(frames_per_sec / target_fps)
            if downsample_factor > 1:
                mel_spec_final = F.avg_pool1d(
                    aligned_mel_spec.unsqueeze(0),
                    kernel_size=downsample_factor,
                    stride=downsample_factor
                ).squeeze(0)
            else:
                mel_spec_final = aligned_mel_spec

        except Exception as e:
            print(f"Error processing raw audio for {vid}: {e}. Returning a zero tensor for mel_spec.")
            target_fps = self.config.audio_config['target_fps']
            expected_mel_frames = int(T * target_fps)
            mel_spec_final = torch.zeros((self.config.audio_config['n_mels'], expected_mel_frames))

        video_frames = []
        try:
            raw_video_path = self.raw_video_paths[idx]
            cap = cv2.VideoCapture(raw_video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video file: {raw_video_path}")

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = self.config.audio_config['target_fps']

            if original_fps <= 0:
                original_fps = 30.0 
                
            sampling_rate = int(round(original_fps / target_fps))
            if sampling_rate < 1: sampling_rate = 1

            for i in range(T):
                frame_index = i * sampling_rate
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()

                if not ret:
                    break

                # OpenCV gives BGR, but ToTensor expects RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_tensor = self.video_transform(frame)
                video_frames.append(frame_tensor)
            
            cap.release()
            
            if video_frames:
                video_frames_tensor = torch.stack(video_frames)
            else:
                raise Exception("No frames were read from video.")

        except Exception as e:
            print(f"Error processing raw video for {vid}: {e}. Returning a zero tensor for video_frames.")
            C = 3
            H = self.frame_resize_dim
            W = self.frame_resize_dim
            video_frames_tensor = torch.zeros((T, C, H, W), dtype=torch.float)

        return {
            "video_id": vid,
            "audio": torch.tensor(audio_feat, dtype=torch.float),
            "visual": torch.tensor(visual_feat, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float),
            "audio_mel_spec": mel_spec_final.to(torch.float),
            "video_frames": video_frames_tensor
        }

class BatchCollator:
    def __init__(self, padding_value: float=0.0):
        self.padding_value = padding_value
    
    def __call__(self, batch):
        video_ids = [d['video_id'] for d in batch]
        audios = [d['audio'] for d in batch]
        visuals = [d['visual'] for d in batch]
        labels = [d['label'] for d in batch]
        audio_mel_specs = [d['audio_mel_spec'] for d in batch]
        video_frames = [d['video_frames'] for d in batch]

        lengths = torch.LongTensor([a.shape[0] for a in audios])
        max_len = int(lengths.max().item())
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        audio_batch = pad_sequence(audios, batch_first=True, padding_value=self.padding_value)
        visual_batch = pad_sequence(visuals, batch_first=True, padding_value=self.padding_value)
        label_batch = pad_sequence(labels, batch_first=True, padding_value=self.padding_value)
        
        padded_mel_specs = pad_sequence([spec.T for spec in audio_mel_specs], batch_first=True, padding_value=self.padding_value)
        mel_spec_batch = padded_mel_specs.permute(0, 2, 1)

        video_frames_batch = pad_sequence(video_frames, batch_first=True, padding_value=self.padding_value)

        return {
            "video_id": video_ids,
            "audio": audio_batch,
            "visual": visual_batch,
            "label": label_batch,
            "mask": mask,
            "audio_mel_spec": mel_spec_batch,
            "video_frames": video_frames_batch
        }

def get_dataloader(config, split: str):
    dataset = AVHDDataset(config, split)
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=(split == "train"),
        collate_fn=BatchCollator(padding_value=0.0),
        pin_memory=True
    )
