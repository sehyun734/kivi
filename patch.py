import types

from torch.nn import Module
import torch
from transformers import PreTrainedModel

from cache import Cache
from models.base import Adapter
from shapes import KV, Hidden, Query


def _repeat_kv(x: KV, n_repeat: int) -> Query:
    if n_repeat == 1:
        return x
    batch_size, n_kv_head, n_tokens, head_dim = x.shape
    return (
        x.unsqueeze(2)  # (batch, n_kv_heads, 1, n_tokens, head_dim)
        .expand(batch_size, n_kv_head, n_repeat, n_tokens, head_dim)
        .flatten(1, 2)  # (batch, n_q_heads(n_kv_heads * n_repeat), n_tokens, head_dim)
    )


def _forward(self: Module, *args, **kwargs) -> tuple[Hidden, None]:
    h: Hidden = kwargs.get("hidden_states", args[0])
    adapter: Adapter = self.adapter
    cache: Cache = kwargs.get("past_key_values")
    layer_i = self.layer_idx
    n_q_heads, n_kv_heads, head_dim = adapter.dims(self)

    q, k, v = adapter.qkv(self, h)
    q, k = adapter.rope(self, q, k, **kwargs)

    n_past_tokens = cache.get_seq_length(layer_i)
    # 메모리병목: attention 계산 순간에는 FP복제본이 생기며 메모리 과다 사용
    k_full, v_full = cache.update(k, v, layer_i)

    # GQA의 경우 kv와 q의 head개수가 다를 수 있음
    k_full = _repeat_kv(k_full, n_q_heads // n_kv_heads)
    v_full = _repeat_kv(v_full, n_q_heads // n_kv_heads)

    n_q_tokens = q.shape[-2]
    n_kv_tokens = k_full.shape[-2]
    scale = adapter.scale(self)
    scores = torch.matmul(q, k_full.transpose(-1, -2)) * scale

    q_pos = torch.arange(n_q_tokens, device=q.device) + n_past_tokens  # 이번 query들의 위치
    k_pos = torch.arange(n_kv_tokens, device=q.device)  # key들의 위치
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # 각 q가 자기보다 미래의 kv를 보지 못하도록
    scores = scores.masked_fill(~causal, float("-inf"))

    attn_per_head = torch.matmul(torch.softmax(scores, dim=-1), v_full)

    attn_out = attn_per_head.transpose(1, 2).reshape(attn_per_head.shape[0], n_q_tokens, n_q_heads * head_dim)
    out = adapter.out(self, attn_out)
    return out, None


def patch(model: PreTrainedModel, adapter: Adapter) -> PreTrainedModel:
    for m in model.modules():
        if adapter.match(m):
            m.adapter = adapter
            m.forward = types.MethodType(_forward, m)
    return model
