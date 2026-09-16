"""
Extracts 3D-ResNet-34 (Kinetics-pretrained) visual features for every TVSum video.

Usage:
    python dataset/preprocess/tvsum_extract_visual_features.py
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from networks.backbones.visual_feature_extractor import VisualFeatureExtractor


def extract_visual_features(video_path, visual_extractor, device="cpu", clip_duration_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frame_clip_len = int(fps * clip_duration_sec)
    if len(frames) < frame_clip_len:
        raise ValueError(f"Too short video (visual): {video_path}")

    visual_feats = []
    for i in range(0, len(frames) - frame_clip_len + 1, frame_clip_len):
        clip = np.array(frames[i:i + frame_clip_len])
        if len(clip) == 0:
            continue

        clip = clip[:16] if len(clip) >= 16 else np.concatenate([clip] + [clip[-1:]] * (16 - len(clip)), axis=0)

        clip_tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        feat = visual_extractor.extract(clip_tensor.to(device))
        visual_feats.append(feat.cpu().numpy())

    if len(visual_feats) == 0:
        raise ValueError(f"No valid visual features extracted: {video_path}")

    return np.stack(visual_feats)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    data_root = os.path.join(os.getenv('DATA_ROOT', './data'), 'tvsum')
    checkpoint_root = os.getenv('CHECKPOINT_ROOT', './checkpoints')
    video_dir = os.path.join(data_root, 'video')
    visual_out_dir = os.path.join(data_root, 'features/visual_emb')
    os.makedirs(visual_out_dir, exist_ok=True)

    visual_extractor = VisualFeatureExtractor(
        checkpoint_path=os.path.join(checkpoint_root, "resnet-34-kinetics.pth"),
        device=device
    )

    video_list = sorted(v for v in os.listdir(video_dir) if v.endswith(".mp4"))
    for vname in tqdm(video_list, desc="Extracting visual features"):
        video_id = os.path.splitext(vname)[0]
        out_path = os.path.join(visual_out_dir, f"{video_id}.npy")
        if os.path.exists(out_path):
            continue
        try:
            feats = extract_visual_features(os.path.join(video_dir, vname), visual_extractor, device)
            np.save(out_path, feats)
            print(f"[{video_id}] Visual shape: {feats.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to process {vname}: {e}")


if __name__ == "__main__":
    main()
