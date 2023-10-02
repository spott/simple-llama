import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from logging import getLogger

from typing import Optional, Tuple

logger = getLogger("__name__")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x/sqrt(avg(x^2) + self.eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # This is transformed to a 32 bit float before being normed in order to 
        # avoid some numerical instability.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (x.shape, freqs_cis.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freq_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    #freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # vector of (1/theta) ^ (i/dim) for i \in 0,2,4,...dim
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # matrix of j*(1/theta)^(i/dim) for i \in 0,2,4,...dim and j \in 0,1,2...max_seq_len*2
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # matrix of e^(j*(1/theta)^(i/dim)) for \in 0,2,4,...dim and j \in 0,1,2...max_seq_len*2 
    return freqs_cis


class Attention(nn.Module):
    def __init__(self, max_batch_size, max_seq_len,  n_heads: int = 32, dim: int = 4096, ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        # the n_heads * head_dim part here is one way that you can shard this across multiple GPUs.
        # each head can theoretically be on a different GPU.

        # query transformation
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        # key transformation
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        # value transformation
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)

        # output transformation
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        # caches to prevent recalc of previous key/value matrix transformations
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
        bsz, seqlen, _ = x.shape
        # apply the transformations:
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # split up the transformed vectors into vectors per head.
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        # we apply the rotary embedding (RoPe) to the keys and query vector.
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # cache_k and cache_v are the keys and values for the entire sequence length.
        # here we are adding the part of the key and value that we are currently handling
        # (start_pos to start_pos + seqlen), so we are doing queries against the entire 
        # sequence, but only have to compute the latest token.
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # the cache_k and cache_v tensors are bigger than needed, so we only look at the part we need.
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        # keys = (batch_size, total_seq_len, n_heads, head_dim)
        vals = self.cache_v[:bsz, : start_pos + seqlen]
        # keys = (batch_size, total_seq_len, n_heads, head_dim)

        # transpose our transformed vectors so that they can be broadcast and multiplied correctly
        query = xq.transpose(1, 2)
        # query = (batch_size, n_heads, seqlen, head_dim) because we run this on each head
        keys = keys.transpose(1, 2)
        # keys = (batch_size, n_heads, head_dim, total_seqlen) because we run this on each head
        vals = vals.transpose(1, 2)
        # vals = (batch_size, n_heads, total_seqlen, head_dim) because we run this on each head

        scores = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # scores = (batch_size, n_heads, seqlen, total_seqlen)
        if mask is not None:
            scores = (
                scores + mask
            )  # add negative infinity to the scores that are from the future.
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # scores is now a probability distribution across every element of the sequence.
        output = torch.matmul(scores, vals)
        # output = (batch_size, n_heads, seqlen, head_dim)
        # and the output is now the values scaled by the scores probability dist.
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # this just drops the "per-head" dim of the output tensor.
        # output = (batch_size, seqlen, dim)
        # and then we apply an output transformation
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[int],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)

        if ffn_dim_multiplier is not None:
            hidden_dim = ffn_dim_multiplier * hidden_dim
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # = 11008 for 7b model... close to 8/3 * dim, but off because of the various rounding.

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # This is a gated feed forward network.  W3 acts as a gate on the nonlinear activation
        # from W1.  See: https://arxiv.org/pdf/2002.05202.pdf.

        # Note that the * represents elementwise multiplication (not matrix multiplication).
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        max_batch_size: int,
        max_seq_len: int,
        dim: int = 4096,
        n_heads: int = 32,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[int] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.n_heads = n_heads
        self.multiple_of = multiple_of
        self.ffn_dim_miltiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
       
        self.attention = Attention(max_batch_size, max_seq_len, n_heads, dim)
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
        # residual connection.  Note we apply the norm before applying attention
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        # another residual connection!  And again, the norm is applied before the ffn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        max_batch_size: int,
        n_layers: int = 32,
        vocab_size: int = 32000,
        dim: int = 4096,
        n_heads: int = 32,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[int] = None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_layers = n_layers              # number of attention/feedforward groups
        self.vocab_size = vocab_size          # number of "words" that the model recognizes.
        self.dim = dim                        # internal representation dimension
        self.n_heads = n_heads                # number of attention heads per attention layer
        self.max_batch_size = max_batch_size  # maximum number of sequences to run at the same time
        self.max_seq_len = max_seq_len        # maximum length of each of those sequences in tokens 
                                              # (elements of the vocab) This includes prompt and
                                              # generated text.
        
        # An embedding can be thought of as a dictionary of size `vocab_size` with a vector of size
        # `dim` for each element.  This is a trainable version of something like Word2Vec.
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # each layer is a "TransformerBlock", which contains an Attention layer, and a Feedforward
        # Layer
        self.layers = nn.ModuleList(
            (
                TransformerBlock(
                    i,
                    max_batch_size,
                    max_seq_len,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for i in range(n_layers)
            )
        )
        
        # This is using RMSNorm, which has a `norm_eps` parameter to prevent divisions by zero.
        self.norm = RMSNorm(dim, norm_eps)

        # A linear layer that inverts the process of the embedding layer.
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # A vector that represents the RoPe complex frequency coefficients
        self.freqs_cis = precompute_freq_cis(dim // n_heads, max_seq_len * 2)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        batch_size, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # we need the relevant precomputed frequency coefficients for our transformer blocks
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        # create a mask to keep the model causal: it shouldn't be able to look to the future.
        # this is irrelevant when we are generating, but is important as we are taking in
        # a sequence.
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h).float() # we convert to 32 bit floats here.

        return output
