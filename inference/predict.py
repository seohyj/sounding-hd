import argparse
import torch
import numpy as np
import os
from types import SimpleNamespace

from networks.ProposedAVHDNet import ProposedAVHDNet

def setup_config(dataset_name):
    return SimpleNamespace(
        audio_config={'n_mels': 128},
        model_config={
            'audio_dim': 2048,
            'visual_dim': 512 if dataset_name == 'tvsum' else 1024,
            'hidden_dim': 512,
            'dropout': 0.1,
            'n_heads': 4,
        },
    )

def load_and_preprocess_features(audio_path, visual_path, melspec_path, device):
    audio_feat = np.load(audio_path)
    visual_feat = np.load(visual_path)
    mel_spec = np.load(melspec_path)

    if audio_feat.ndim == 3:
        audio_feat = audio_feat.squeeze(0)

    min_len = min(len(audio_feat), len(visual_feat))
    
    audio_feat = audio_feat[:min_len]
    visual_feat = visual_feat[:min_len]

    expected_mel_len = min_len
    
    if mel_spec.shape[1] > expected_mel_len:
        mel_spec = mel_spec[:, :expected_mel_len]
    elif mel_spec.shape[1] < expected_mel_len:
        padding_len = expected_mel_len - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, padding_len)), 'constant', constant_values=0)

    audio_tensor = torch.from_numpy(audio_feat).float().to(device)
    visual_tensor = torch.from_numpy(visual_feat).float().to(device)
    mel_tensor = torch.from_numpy(mel_spec).float().to(device)

    return (
        audio_tensor.unsqueeze(0),
        visual_tensor.unsqueeze(0),
        mel_tensor.unsqueeze(0)
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run a trained DAViHD checkpoint over a directory of pre-extracted features.")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to best_model.pt produced by main.py")
    parser.add_argument('--audio_feat_dir', type=str, required=True, help="Directory of PANN audio embedding .npy files")
    parser.add_argument('--visual_feat_dir', type=str, required=True, help="Directory of visual embedding .npy files")
    parser.add_argument('--melspec_dir', type=str, required=True, help="Directory of log-mel spectrogram .npy files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to write predicted score .npy files")
    parser.add_argument('--dataset', type=str, default='mrhisum', choices=['mrhisum', 'tvsum'])
    return parser.parse_args()

def main():
    args = parse_args()
    CKPT_PATH = args.ckpt_path
    AUDIO_FEAT_PATH = args.audio_feat_dir
    VISUAL_FEAT_PATH = args.visual_feat_dir
    MELSPEC_FEAT_PATH = args.melspec_dir
    OUTPUT_DIR = args.output_dir
    DATASET = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = setup_config(DATASET)

    model = ProposedAVHDNet(config)
    
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    video_ids = [f.split('.')[0] for f in os.listdir(AUDIO_FEAT_PATH) if f.endswith('.npy')]
    
    print(f"Found {len(video_ids)} videos to process. Starting inference...")

    for video_id in video_ids:
        audio_path = os.path.join(AUDIO_FEAT_PATH, f'{video_id}.npy')
        visual_path = os.path.join(VISUAL_FEAT_PATH, f'{video_id}.npy')
        melspec_path = os.path.join(MELSPEC_FEAT_PATH, f'{video_id}.npy')

        audio_tensor, visual_tensor, mel_tensor = load_and_preprocess_features(
            audio_path, visual_path, melspec_path, device
        )

        with torch.no_grad():
            predicted_scores = model(
                visual_feat=visual_tensor,
                audio_feat=audio_tensor,
                audio_mel_spec=mel_tensor
            )
        
        scores_numpy = predicted_scores.squeeze(0).cpu().numpy()

        output_path = os.path.join(OUTPUT_DIR, f'{video_id}.npy')
        np.save(output_path, scores_numpy)

if __name__ == "__main__":
    main()