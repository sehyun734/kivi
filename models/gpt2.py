from torch.nn import Module

from cache import Cache
from models.base import Adapter, register
from shapes import KV, Query, Hidden


@register("openai-community/gpt2")
class GPT2(Adapter):
    def __init__(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

        self.attn = GPT2Attention

    def cache(self, **kwargs) -> Cache:
        raise NotImplementedError

    def layer_i(self, m: Module) -> int:
        raise NotImplementedError

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

    def out(self, m: Module, h: Hidden) -> Hidden:
        return m.c_proj(h)

    def rope(self, m: Module, q: Query, k: KV, **kwargs) -> tuple[Query, KV]:
        return q, k
