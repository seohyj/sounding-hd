import torch.nn as nn


class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, return_attention=False):
        key_padding_mask = None
        if mask is not None:
            mask = mask.bool()
            key_padding_mask = ~mask

        if return_attention:
            x2, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False,
                                         key_padding_mask=key_padding_mask)
        else:
            x2, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)

        x = self.norm1(x + x2)
        x = self.norm2(x + self.ffn(x))

        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return (x, attn_weights) if return_attention else x


class CrossModalAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, n_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_kv, dim_out)
        self.v_proj = nn.Linear(dim_kv, dim_out)
        self.attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=n_heads, batch_first=True)

    def forward(self, q, kv, q_mask=None, kv_mask=None, return_attention=False):
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask.bool()

        if return_attention:
            out, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=False,
                                          key_padding_mask=key_padding_mask)
        else:
            out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)

        if q_mask is not None:
            out = out.masked_fill(~q_mask.bool().unsqueeze(-1), 0.0)

        return (out, attn_weights) if return_attention else out
