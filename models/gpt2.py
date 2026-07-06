from torch import Module

from models.base import Adapter
from common.shapes import KV, Query, Hidden


class GPT2(Adapter):
    """
    GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
            (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
            (attn): GPT2Attention(
                (c_attn): Conv1D(nf=2304, nx=768)
                (c_proj): Conv1D(nf=768, nx=768)
                (attn_dropout): Dropout(p=0.1, inplace=False)
                (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
            (mlp): GPT2MLP(
                (c_fc): Conv1D(nf=3072, nx=768)
                (c_proj): Conv1D(nf=768, nx=3072)
                (act): NewGELUActivation()
                (dropout): Dropout(p=0.1, inplace=False)
            )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True, bias=True)
    )
    """

    def __init__(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        self.attn = GPT2Attention

    def dims(self, m: Module) -> tuple[int, int, int]:
        return m.num_heads, m.num_heads, m.head_dim

    def qkv(self, m: Module, h: Hidden) -> tuple[Query, KV, KV]:
        batch, n_tokens, _ = h.shape
        n_heads, _, head_dim = self.dims(m)
        q, k, v = m.c_attn(h).split(m.split_size, dim=2)

        def split_heads(x: Hidden) -> KV:
            return x.view(batch, n_tokens, n_heads, head_dim).permute(0, 2, 1, 3)

        return split_heads(q), split_heads(k), split_heads(v)

    def scale(self, m: Module) -> float:
        return m.scaling

    def out(self, m: Module, x: Hidden) -> Hidden:
        return m.c_proj(x)
