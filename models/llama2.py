from torch.nn import Module

from cache import Cache
from models.base import Adapter
from shapes import KV, Query, Hidden


class Llama2(Adapter):
    def __init__(self):
        from transformers.models.llama.modeling_llama import LlamaAttention

        self.attn = LlamaAttention

    def cache(self, **kwargs) -> Cache:
        return kwargs.get("past_key_value")

    def layer_i(self, m: Module) -> int:
        return m.layer_idx

    def dims(self, m: Module) -> tuple[int, int, int]:
        return (
            m.config.num_attention_heads,
            m.config.num_key_value_heads,
            m.head_dim,
        )

    def qkv(self, m: Module, h: Hidden) -> tuple[Query, KV, KV]:
        batch_size, n_tokens, _ = h.shape
        shape = (batch_size, n_tokens, -1, m.head_dim)
        q = m.q_proj(h).view(shape).transpose(1, 2)
        k = m.k_proj(h).view(shape).transpose(1, 2)
        v = m.v_proj(h).view(shape).transpose(1, 2)
        return q, k, v

    def out(self, m: Module, h: Hidden) -> Hidden:
        return m.o_proj(h)

    def scale(self, m: Module) -> float:
        return m.scaling

    def rope(self, m: Module, q: Query, k: KV, **kwargs) -> tuple[Query, KV]:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        pos_embed = kwargs.get("position_embeddings")
        if pos_embed is None:
            return q, k
        cos, sin = pos_embed
        return apply_rotary_pos_emb(q, k, cos, sin)
