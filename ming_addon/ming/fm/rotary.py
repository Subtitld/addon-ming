"""The two rotary-embedding helpers the flow head takes from x-transformers.

Copied from x-transformers 2.9.2 (MIT, Phil Wang), the version Ming-3.0-tts
pins, without the xpos and interpolation options the head does not use, and
with einops replaced by reshapes. Saves bundling x-transformers and its
dependencies for some twenty lines.
"""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    def forward(self, t):
        # Always float32, as upstream's disables autocast here.
        if t.ndim == 1:
            t = t.unsqueeze(0)
        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)   # interleaved: (d r)
        return freqs, 1.


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim=-1).type(orig_dtype)
