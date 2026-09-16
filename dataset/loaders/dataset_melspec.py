
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class AVHDDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.dataset = config.dataset
        self.split = split

        self.audio_emb_dir = config.audio_dir
        self.visual_emb_dir = config.visual_dir
        self.label_dir = config.label_dir
        self.mel_spec_dir = config.mel_spec_dir

        if not self.mel_spec_dir or not os.path.exists(self.mel_spec_dir):
            raise ValueError(f"Mel-spectrogram directory not provided or not found: {self.mel_spec_dir}")

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

        self.video_ids = self._filter_existing_files()
        
        print(f"[{split.upper()}] Using pre-computed Mel-Spectrograms.")
        print(f"[{split.upper()}] Total usable videos: {len(self.video_ids)}")

    def _filter_existing_files(self):
        valid_ids = []
        for vid in self.video_ids:
            audio_emb_path = os.path.join(self.audio_emb_dir, f"{vid}.npy")
            visual_emb_path = os.path.join(self.visual_emb_dir, f"{vid}.npy")
            label_path = os.path.join(self.label_dir, f"{vid}.npy")
            mel_spec_path = os.path.join(self.mel_spec_dir, f"{vid}.npy")

            if (os.path.exists(audio_emb_path) and
                os.path.exists(visual_emb_path) and
                os.path.exists(label_path) and
                os.path.exists(mel_spec_path)):
                valid_ids.append(vid)
        return valid_ids

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

        audio_feat = np.load(os.path.join(self.audio_emb_dir, f"{vid}.npy"))
        visual_feat = np.load(os.path.join(self.visual_emb_dir, f"{vid}.npy"))
        label = np.load(os.path.join(self.label_dir, f"{vid}.npy"))

        if audio_feat.ndim == 3:
            audio_feat = audio_feat.squeeze(0)

        T = min(len(audio_feat), len(visual_feat), len(label))
        
        audio_feat = audio_feat[:T]
        visual_feat = visual_feat[:T]
        label = label[:T]

        mel_spec_path = os.path.join(self.mel_spec_dir, f"{vid}.npy")
        try:
            mel_spec_final = torch.from_numpy(np.load(mel_spec_path))
            
            expected_len = T * 1
            if mel_spec_final.shape[1] > expected_len:
                mel_spec_final = mel_spec_final[:, :expected_len]
            elif mel_spec_final.shape[1] < expected_len:
                padding = expected_len - mel_spec_final.shape[1]
                mel_spec_final = F.pad(mel_spec_final, (0, padding), 'constant', 0)

        except Exception as e:
            print(f"Error loading pre-computed mel-spec for {vid}: {e}. Returning zero tensor.")
            expected_mel_frames = T * 1
            mel_spec_final = torch.zeros((self.config.audio_config['n_mels'], expected_mel_frames))

        return {
            "video_id": vid,
            "audio": torch.tensor(audio_feat, dtype=torch.float),
            "visual": torch.tensor(visual_feat, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float),
            "audio_mel_spec": mel_spec_final.to(torch.float)
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

        lengths = torch.LongTensor([a.shape[0] for a in audios])
        max_len = int(lengths.max().item())
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        audio_batch = pad_sequence(audios, batch_first=True, padding_value=self.padding_value)
        visual_batch = pad_sequence(visuals, batch_first=True, padding_value=self.padding_value)
        label_batch = pad_sequence(labels, batch_first=True, padding_value=self.padding_value)
        
        padded_mel_specs = pad_sequence([spec.T for spec in audio_mel_specs], batch_first=True, padding_value=self.padding_value)
        mel_spec_batch = padded_mel_specs.permute(0, 2, 1)

        return {
            "video_id": video_ids,
            "audio": audio_batch,
            "visual": visual_batch,
            "label": label_batch,
            "mask": mask,
            "audio_mel_spec": mel_spec_batch
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
