import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.tapconv.tfd_conv import DYCNN


class AdvancedAudioEncoder(nn.Module):
    """Audio dynamics encoder: frequency-dynamic convolution over the log-mel spectrogram,
    resampled to the visual frame rate and contextualised by a Transformer."""

    # Time pooling inside the CNN, so a cropped sequence still needs this many frames.
    _MIN_MEL_FRAMES = 4

    def __init__(self, n_input_ch=1, embed_dim=512, n_transformer_layer=2, dycnn_kwargs=None):
        super().__init__()
        self.embed_dim = embed_dim

        dycnn_config = {
            'n_filt': [16, 32, 64, 128, 128, 128],
            'kernel': [3] * 6, 'pad': [1] * 6, 'stride': [1] * 6,
            'pooling': [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2), (1, 2)],
            'DY_layers': [0, 1, 1, 1, 1, 1],
            'pool_dim': 'time',
            'pool_type': 'SE2D_xf_all',
            'n_mels': 128,
        }
        if dycnn_kwargs:
            dycnn_config.update(dycnn_kwargs)

        self.tfd_conv = DYCNN(n_input_ch=n_input_ch, **dycnn_config)

        cnn_out_freq_dim = dycnn_config['n_mels']
        for _, freq_pool in dycnn_config['pooling']:
            cnn_out_freq_dim = math.ceil(cnn_out_freq_dim / freq_pool)
        cnn_flat_dim = self.tfd_conv.n_filt_last * cnn_out_freq_dim

        self.feature_projection = nn.Linear(cnn_flat_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='relu', batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layer)
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, dropout=0.1)

    def _encode_one(self, mel_spec, out_len):
        """Encode a batch whose temporal extent is entirely valid (no padding)."""
        x = mel_spec.unsqueeze(1).permute(0, 1, 3, 2)   # (B, F, T) -> (B, 1, T, F)
        x = self.tfd_conv(x)                            # (B, C, T', F')
        x = x.permute(0, 1, 3, 2).flatten(1, 2)         # (B, C*F', T')

        if x.size(2) != out_len:
            x = F.adaptive_avg_pool1d(x, output_size=out_len)

        x = self.feature_projection(x.permute(0, 2, 1))
        x = self.positional_encoding(x)
        return self.transformer_encoder(x)

    def forward(self, mel_spec, out_len, mask=None):
        """
        mel_spec: (B, n_mels, T) log-mel spectrogram
        out_len:  padded temporal length shared by the rest of the batch
        mask:     (B, T), True on real frames

        The convolutions, the temporal attention pooling inside the TFD conv and the adaptive
        pooling all read the whole time axis, so running them over a batch-padded tensor would
        let one video's length change another's representation. Each sample is therefore
        encoded from its own valid extent and padded back afterwards.
        """
        if mask is None:
            return self._encode_one(mel_spec, out_len)

        mask = mask.bool()
        lengths = mask.sum(dim=1)
        max_mel_frames = mel_spec.size(2)

        if bool((lengths >= max_mel_frames).all()):
            return self._encode_one(mel_spec, out_len)

        outputs = None
        for i in range(mel_spec.size(0)):
            length = max(min(int(lengths[i].item()), max_mel_frames), 1)
            sample = mel_spec[i:i + 1, :, :length]
            if length < self._MIN_MEL_FRAMES:
                sample = F.pad(sample, (0, self._MIN_MEL_FRAMES - length), mode='replicate')

            encoded = self._encode_one(sample, length)
            if outputs is None:
                outputs = encoded.new_zeros(mel_spec.size(0), out_len, encoded.size(-1))
            valid = min(length, out_len)
            outputs[i, :valid] = encoded[0, :valid]

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.dropout(x + self.pe[:x.size(0), :])
        return x.permute(1, 0, 2)


class AudioFeatureFuser(nn.Module):
    """Gates the semantic stream with the dynamics stream (element-wise multiplication)."""

    def __init__(self, embed_dim=512):
        super().__init__()

    def forward(self, semantic_features, dynamic_features):
        min_len = min(semantic_features.size(1), dynamic_features.size(1))
        return semantic_features[:, :min_len, :] * dynamic_features[:, :min_len, :]
