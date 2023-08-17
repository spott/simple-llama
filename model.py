import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from logging import getLogger

from typing import Optional, Tuple

logger = getLogger("__name__")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 1/sqrt(avg(x^2) + self.eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    #logger.info(f"reshape_for_broadcast: {freqs_cis.shape}, {x.shape}")
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (x.shape, freqs_cis.shape)
    # (1, x.shape[1], 1s, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    #logger.info(f"apply_rotary_emb: {xq.shape}, {xk.shape}, {freqs_cis.shape}")
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freq_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    #logger.info(f"precomute_freq_cis: {dim}, {end}, {theta}")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


class Attention(nn.Module):
    """
    x = input tensor in shape: batch_size, sequence_length, embedding_dim
    wq, wk, wv = Learned Matrices of shape dim->dim for taking x to query, key, and value vectors
    cache_k, cache_v = cache of the key and value so that previous positions are taken into account.
    freqs_cis = vector that represents the rotary embedding...
    mask = triangular matrix for masking out the future.

    """

    def __init__(self, n_heads, dim, max_batch_size, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, n_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, n_heads, self.head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        #logger.info(f"Attention.forward: {x.shape}, {start_pos}, {freqs_cis.shape}, {mask.shape if mask is not None else None}")
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        vals = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        vals = vals.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = (
                scores + mask
            )  # add negative infinity to the scores that are from the future.
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, vals)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) + self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        multiple_of: int,
        ffn_dim_multiplier,
        norm_eps: float,
        max_batch_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.multiple_of = multiple_of
        self.ffn_dim_miltiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.attention = Attention(n_heads, dim, max_batch_size, max_seq_len)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.feed_forward = FeedForward(dim, 4 * dim, multiple_of, ffn_dim_multiplier)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        # residual connection:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        # another residual connection!
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        dim: int,
        n_heads: int,
        multiple_of: int,
        ffn_dim_multiplier,
        norm_eps: float,
        max_seq_len: int,
        max_batch_size: int,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList(
            (
                TransformerBlock(
                    i,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    max_batch_size,
                    max_seq_len,
                )
                for i in range(n_layers)
            )
        )

        self.norm = RMSNorm(dim, norm_eps)

        self.output = nn.Linear(dim, vocab_size, bias=False)

        self.freqs_cis = precompute_freq_cis(dim // n_heads, max_seq_len * 2)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            #logger.debug(f"h: {h[:,:,:30]}")
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        #logger.debug(f"h: {h[:,:,:30]}")
        output = self.output(h).float()
        #logger.debug(f"output: {output[:,:,:30]}")
        return output
