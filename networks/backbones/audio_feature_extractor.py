import torch
import torch.nn as nn
import torchaudio
from networks.backbones.PANN import Cnn14_nontemp

def preprocess_audio(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform

class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, checkpoint_path, device="cpu"):
        super().__init__()
        self.device = device

        self.audio_model = Cnn14_nontemp(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=527
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.audio_model.load_state_dict(checkpoint['model'])

        self.audio_model.to(device)
        self.audio_model.eval()
        for param in self.audio_model.parameters():
            param.requires_grad = False

    def forward(self, waveform):
        with torch.no_grad():
            if waveform.dim() == 3:
                B, T, L = waveform.shape
                waveform = waveform.view(B * T, L)
                embedding = self.audio_model(waveform)['embedding']
                embedding = embedding.view(B, T, -1)
            else:
                embedding = self.audio_model(waveform)['embedding']
        return embedding