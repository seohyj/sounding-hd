"""
Extracts PANN audio embeddings for every video in a dataset, in 1-second chunks,
matching the paper's audio semantic feature extraction. torchaudio decodes each
video's audio track directly via its ffmpeg backend (see README Requirements).

Usage:
    python dataset/preprocess/extract_audio_features.py --dataset tvsum
    python dataset/preprocess/extract_audio_features.py --dataset mrhisum
"""

import argparse
import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from networks.backbones.audio_feature_extractor import AudioFeatureExtractor

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.webm', '.avi')


def extract_audio_features(video_path, audio_extractor, device="cpu", clip_duration_sec=1):
    waveform, sr = torchaudio.load(video_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    total_samples = waveform.shape[1]
    clip_samples = int(16000 * clip_duration_sec)
    num_clips = (total_samples + clip_samples - 1) // clip_samples
    pad_len = num_clips * clip_samples - total_samples
    if pad_len > 0:
        waveform = torch.cat([waveform, torch.zeros(1, pad_len)], dim=1)

    audio_chunks = waveform.view(1, num_clips, clip_samples)
    with torch.no_grad():
        audio_feat = audio_extractor(audio_chunks.to(device))
    return audio_feat.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Extract PANN audio embeddings.")
    parser.add_argument('--dataset', type=str, required=True, choices=['tvsum', 'mrhisum'])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    data_root = os.path.join(os.getenv('DATA_ROOT', './data'), args.dataset)
    checkpoint_root = os.getenv('CHECKPOINT_ROOT', './checkpoints')
    video_dir = os.path.join(data_root, 'video')
    audio_out_dir = os.path.join(data_root, 'audio_emb' if args.dataset == 'mrhisum' else 'features/audio_emb')
    os.makedirs(audio_out_dir, exist_ok=True)

    audio_extractor = AudioFeatureExtractor(
        sample_rate=16000,
        window_size=512,
        hop_size=160,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        checkpoint_path=os.path.join(checkpoint_root, "Cnn14_16k_mAP=0.438.pth"),
        device=device
    )

    video_files = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(VIDEO_EXTENSIONS))
    for fname in tqdm(video_files, desc="Extracting audio features"):
        video_id = os.path.splitext(fname)[0]
        out_path = os.path.join(audio_out_dir, f"{video_id}.npy")
        if os.path.exists(out_path):
            continue
        try:
            feat = extract_audio_features(os.path.join(video_dir, fname), audio_extractor, device)
            np.save(out_path, feat)
        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")


if __name__ == "__main__":
    main()
