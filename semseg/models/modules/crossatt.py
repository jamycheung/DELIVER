import torch
from torch import nn
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True)
    return t.softmax(dim = dim)

# bidirectional cross attention - have two sequences attend to each other with 1 attention step

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, context_dim=None, dropout=0., talking_heads=False, prenorm=True,):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(self, x, context, return_attn=False, rel_pos_bias=None):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # get similarities
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # relative positional bias, if supplied
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # get attention along both sequence length and context length dimensions
        # shared similarity matrix
        attn = stable_softmax(sim, dim = -1)
        context_attn = stable_softmax(sim, dim = -2)

        # dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates values from src sequence
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # merge heads and combine out
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out

if __name__ == '__main__':
    video = torch.randn(1, 4096, 512)
    audio = torch.randn(1, 8192, 386)

    joint_cross_attn = BidirectionalCrossAttention(dim = 512, heads = 8, dim_head = 64, context_dim = 386)
    video_out, audio_out = joint_cross_attn(video, audio)

    # attended output should have the same shape as input
    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape