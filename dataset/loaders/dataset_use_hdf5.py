# HDF5 format dataset for improved I/O performance

import os
import json
import torch
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

def worker_init_fn(worker_id):
    """
    DataLoader worker init function. Each worker opens and owns its own HDF5 file handles.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._open_hdf5_files()  # Let each worker open its own independent file handles

class AVHDDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.dataset = config.dataset
        self.split = split

        data_root = os.getenv('MRHISUM_HDF5_ROOT', os.path.join(os.getenv('DATA_ROOT', './data'), 'mrhisum'))
        self.audio_hdf5_path = os.path.join(data_root, "audio_features.h5")
        self.visual_hdf5_path = os.path.join(data_root, "visual_features.h5")
        self.melspec_hdf5_path = os.path.join(data_root, "melspec_features.h5")
        self.gtscore_hdf5_path = os.path.join(data_root, "gtscore.h5")

        required_files = [
            self.audio_hdf5_path, self.visual_hdf5_path, 
            self.melspec_hdf5_path, self.gtscore_hdf5_path
        ]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        if self.dataset == "tvsum":
            self.video_ids = self.create_tvsum_split(config.seed)
        elif self.dataset == "mrhisum":
            with open(config.split_file, "r") as f:
                splits = json.load(f)
            self.video_ids = splits[f"{split}_keys"]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # File handles start as None; each worker opens its own via worker_init_fn.
        self.audio_h5 = None
        self.visual_h5 = None
        self.melspec_h5 = None
        self.gtscore_h5 = None
        
        self.video_ids = self._filter_existing_videos()
        
        print(f"[{split.upper()}] Using HDF5 format.")
        print(f"[{split.upper()}] Total usable videos: {len(self.video_ids)}")

    def _open_hdf5_files(self):
        self.audio_h5 = h5py.File(self.audio_hdf5_path, 'r')
        self.visual_h5 = h5py.File(self.visual_hdf5_path, 'r')
        self.gtscore_h5 = h5py.File(self.gtscore_hdf5_path, 'r')
        self.melspec_h5 = h5py.File(self.melspec_hdf5_path, 'r')

    def _filter_existing_videos(self):
        valid_ids = []
        
        available_videos = {}
        hdf5_files = {
            'audio': self.audio_hdf5_path,
            'visual': self.visual_hdf5_path, 
            'melspec': self.melspec_hdf5_path,
            'gtscore': self.gtscore_hdf5_path
        }
        
        for name, file_path in hdf5_files.items():
            with h5py.File(file_path, 'r') as hf:
                available_videos[name] = set(hf.keys())
        
        common_videos = set.intersection(*available_videos.values())
        
        for vid in self.video_ids:
            if vid in common_videos:
                valid_ids.append(vid)
        
        missing_count = len(self.video_ids) - len(valid_ids)
        if missing_count > 0:
            print(f"Warning: {missing_count} videos missing from HDF5 files")
        
        return valid_ids

    def create_tvsum_split(self, seed=42, split_ratio=0.8):
        with h5py.File(self.audio_hdf5_path, 'r') as hf:
            all_ids = sorted(list(hf.keys()))
        
        train_ids, test_ids = train_test_split(all_ids, train_size=split_ratio, random_state=seed)
        if self.split == "train":
            return train_ids
        else:
            return test_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        # Fallback for num_workers=0, where worker_init_fn never runs
        if self.audio_h5 is None:
            self._open_hdf5_files()

        vid = self.video_ids[idx]

        try:
            audio_feat = self.audio_h5[vid][:]
            visual_feat = self.visual_h5[vid][:]
            label = self.gtscore_h5[vid][:]
            mel_spec_final = self.melspec_h5[vid][:]

        except Exception as e:
            print(f"Error loading {vid} from HDF5: {str(e)}")
            audio_feat = np.zeros((100, 2048), dtype=np.float32)
            visual_feat = np.zeros((100, 1024), dtype=np.float32)
            label = np.zeros(100, dtype=np.float32)
            mel_spec_final = np.zeros((64, 1000), dtype=np.float32)

        if audio_feat.ndim == 3:
            audio_feat = audio_feat.squeeze(0)

        T = min(len(audio_feat), len(visual_feat), len(label))
        
        audio_feat = audio_feat[:T]
        visual_feat = visual_feat[:T]
        label = label[:T]

        expected_len = T * 1
        if mel_spec_final.shape[1] > expected_len:
            mel_spec_final = mel_spec_final[:, :expected_len]
        elif mel_spec_final.shape[1] < expected_len:
            padding = expected_len - mel_spec_final.shape[1]
            mel_spec_final = np.pad(mel_spec_final, ((0, 0), (0, padding)), 'constant', constant_values=0)

        return {
            "video_id": vid,
            "audio": torch.tensor(audio_feat, dtype=torch.float),
            "visual": torch.tensor(visual_feat, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float),
            "audio_mel_spec": torch.tensor(mel_spec_final, dtype=torch.float)
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
        pin_memory=True,
        worker_init_fn=worker_init_fn if config.num_workers > 0 else None
    )