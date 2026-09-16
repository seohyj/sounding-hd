"""
Generates log-mel spectrograms (for the Audio Dynamics Encoder) for every video in a
dataset, downsampled to 1fps to align with the PANN semantic features. torchaudio
decodes each video's audio track directly via its ffmpeg backend.

Usage:
    python dataset/preprocess/extract_melspec.py --dataset tvsum
    python dataset/preprocess/extract_melspec.py --dataset mrhisum
"""

import argparse
import os
import json
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.webm', '.avi')

AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'win_length': 2048,
    'hop_length': 256,
    'n_mels': 128,
    'target_fps': 1,
}


def find_video_file(vid, video_dir):
    for ext in VIDEO_EXTENSIONS:
        path = os.path.join(video_dir, vid + ext)
        if os.path.exists(path):
            return path
    return None


def get_video_ids(dataset, data_root, feature_dir):
    if dataset == 'tvsum':
        return sorted(os.path.splitext(f)[0] for f in os.listdir(feature_dir) if f.endswith('.npy'))
    else:
        split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mrhisum_split.json')
        with open(split_path, 'r') as f:
            splits = json.load(f)
        return sorted(set(splits.get('train_keys', []) + splits.get('val_keys', []) + splits.get('test_keys', [])))


def extract_melspec(video_dir, feature_dir, output_dir, video_ids):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Source video directory: {video_dir}")
    print(f"Source feature directory: {feature_dir}")
    print(f"Output directory for Mel-spectrograms: {output_dir}")
    print(f"Found {len(video_ids)} video IDs to process")

    mel_transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=AUDIO_CONFIG['sample_rate'],
        n_fft=AUDIO_CONFIG['n_fft'],
        win_length=AUDIO_CONFIG['win_length'],
        hop_length=AUDIO_CONFIG['hop_length'],
        n_mels=AUDIO_CONFIG['n_mels']
    ).to(torch.device('cpu'))

    for vid in tqdm(video_ids, desc="Processing videos"):
        try:
            feature_path = os.path.join(feature_dir, f"{vid}.npy")
            if not os.path.exists(feature_path):
                print(f"\nSkipping {vid}: Missing audio feature file at {feature_path}.")
                continue
            T = len(np.load(feature_path))

            video_path = find_video_file(vid, video_dir)
            if not video_path:
                print(f"\nSkipping {vid}: Raw video file not found in {video_dir}.")
                continue

            waveform, sr = torchaudio.load(video_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != AUDIO_CONFIG['sample_rate']:
                waveform = torchaudio.transforms.Resample(sr, AUDIO_CONFIG['sample_rate'])(waveform)

            mel_spec = (mel_transformer(waveform).squeeze(0) + 1e-8).log()
            frames_per_sec = AUDIO_CONFIG['sample_rate'] / AUDIO_CONFIG['hop_length']
            expected_mel_frames = int(T * frames_per_sec)

            current_mel_frames = mel_spec.shape[1]
            if current_mel_frames > expected_mel_frames:
                mel_spec = mel_spec[:, :expected_mel_frames]
            elif current_mel_frames < expected_mel_frames:
                mel_spec = F.pad(mel_spec, (0, expected_mel_frames - current_mel_frames), 'constant', 0)

            target_fps = AUDIO_CONFIG['target_fps']
            downsample_factor = int(frames_per_sec / target_fps)
            if downsample_factor > 1:
                mel_spec = F.avg_pool1d(mel_spec.unsqueeze(0), kernel_size=downsample_factor, stride=downsample_factor).squeeze(0)

            output_path = os.path.join(output_dir, f"{vid}.npy")
            np.save(output_path, mel_spec.cpu().numpy())

        except Exception as e:
            print(f"\n[ERROR] Could not process video {vid}. Reason: {e}. Skipping.")

    print(f"\nPreprocessing finished. Saved results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract log-mel spectrograms.")
    parser.add_argument('--dataset', type=str, required=True, choices=['tvsum', 'mrhisum'])
    args = parser.parse_args()

    data_root = os.path.join(os.getenv('DATA_ROOT', './data'), args.dataset)
    video_dir = os.path.join(data_root, 'video')
    if args.dataset == 'tvsum':
        feature_dir = os.path.join(data_root, 'features/audio_emb')
        output_dir = os.path.join(data_root, 'features/melspec')
    else:
        feature_dir = os.path.join(data_root, 'audio_emb')
        output_dir = os.path.join(data_root, 'audio/melspec')

    video_ids = get_video_ids(args.dataset, data_root, feature_dir)
    extract_melspec(video_dir, feature_dir, output_dir, video_ids)


if __name__ == "__main__":
    main()
