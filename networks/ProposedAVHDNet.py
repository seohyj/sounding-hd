import torch
import torch.nn as nn

from networks.audio_visual_net import SelfAttentionEncoder, CrossModalAttention
from networks.proposed_audio_net import AdvancedAudioEncoder, AudioFeatureFuser


class ProposedAVHDNet(nn.Module):
    """DAViHD: a visual encoder and a dual-pathway audio encoder, fused by bidirectional
    cross-modal attention into a per-frame highlight score."""

    def __init__(self, config):
        super().__init__()

        audio_dim = config.model_config['audio_dim']
        visual_dim = config.model_config['visual_dim']
        hidden_dim = config.model_config['hidden_dim']
        n_heads = config.model_config['n_heads']
        dropout = config.model_config['dropout']

        self.visual_encoder = SelfAttentionEncoder(dim=visual_dim, n_heads=n_heads, dropout=dropout)

        # Audio semantic pathway: self-attention over the PANNs embeddings.
        self.semantic_audio_encoder = SelfAttentionEncoder(dim=audio_dim, n_heads=n_heads, dropout=dropout)

        # Audio dynamics pathway: frequency-dynamic convolution over the log-mel spectrogram.
        self.dynamic_audio_encoder = AdvancedAudioEncoder(
            embed_dim=audio_dim,
            n_transformer_layer=2,
            dycnn_kwargs={'n_mels': config.audio_config['n_mels']},
        )

        # Early-SA fusion: each stream is contextualised before they are multiplied.
        self.audio_fuser = AudioFeatureFuser(embed_dim=audio_dim)

        self.cross_audio_to_visual = CrossModalAttention(dim_q=audio_dim, dim_kv=visual_dim,
                                                         dim_out=audio_dim, n_heads=n_heads)
        self.cross_visual_to_audio = CrossModalAttention(dim_q=visual_dim, dim_kv=audio_dim,
                                                         dim_out=visual_dim, n_heads=n_heads)

        fusion_input_dim = 2 * (visual_dim + audio_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.fusion_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, visual_feat, audio_feat, audio_mel_spec, mask=None, return_attention=False):
        """
        visual_feat:    (B, T, D_vis)
        audio_feat:     (B, T, D_aud) PANNs semantic embeddings
        audio_mel_spec: (B, n_mels, T) log-mel spectrogram
        mask:           (B, T), True on real frames
        returns:        (B, T) highlight score in [0, 1]
        """
        if mask is not None:
            mask = mask.bool()

        attention_weights = {}

        if return_attention:
            v_encoded, visual_self_attn = self.visual_encoder(visual_feat, mask=mask, return_attention=True)
            a_semantic, audio_self_attn = self.semantic_audio_encoder(audio_feat, mask=mask, return_attention=True)
            attention_weights['visual_self_attention'] = visual_self_attn
            attention_weights['audio_self_attention'] = audio_self_attn
        else:
            v_encoded = self.visual_encoder(visual_feat, mask=mask)
            a_semantic = self.semantic_audio_encoder(audio_feat, mask=mask)

        a_dynamic = self.dynamic_audio_encoder(audio_mel_spec, out_len=audio_feat.size(1), mask=mask)
        a_encoded = self.audio_fuser(a_semantic, a_dynamic)

        if mask is not None:
            pad = ~mask.unsqueeze(-1)
            v_encoded = v_encoded.masked_fill(pad, 0.0)
            a_encoded = a_encoded.masked_fill(pad, 0.0)

        if return_attention:
            v_cross, v2a_attn = self.cross_visual_to_audio(v_encoded, a_encoded, q_mask=mask,
                                                           kv_mask=mask, return_attention=True)
            a_cross, a2v_attn = self.cross_audio_to_visual(a_encoded, v_encoded, q_mask=mask,
                                                           kv_mask=mask, return_attention=True)
            attention_weights['visual_to_audio'] = v2a_attn
            attention_weights['audio_to_visual'] = a2v_attn
        else:
            v_cross = self.cross_visual_to_audio(v_encoded, a_encoded, q_mask=mask, kv_mask=mask)
            a_cross = self.cross_audio_to_visual(a_encoded, v_encoded, q_mask=mask, kv_mask=mask)

        v_cross = v_cross + v_encoded
        a_cross = a_cross + a_encoded

        if mask is not None:
            pad = ~mask.unsqueeze(-1)
            v_cross = v_cross.masked_fill(pad, 0.0)
            a_cross = a_cross.masked_fill(pad, 0.0)
            v_encoded = v_encoded.masked_fill(pad, 0.0)
            a_encoded = a_encoded.masked_fill(pad, 0.0)

        fusion_input = torch.cat([v_encoded, a_encoded, v_cross, a_cross], dim=-1)
        score = self.fusion_mlp(fusion_input).squeeze(-1)

        if return_attention:
            return score, attention_weights
        return score
