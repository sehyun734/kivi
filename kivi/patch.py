import types

from torch.nn import Module
import torch
from transformers import PreTrainedModel

from kivi.cache import Cache
from models.base import Adapter
from common.shapes import KV, Hidden, Query


def _repeat_kv(x: KV, n_repeat: int) -> Query:
    if n_repeat == 1:
        return x
    batch_size, n_kv_head, n_tokens, head_dim = x.shape
    return (
        x.unsqueeze(2)  # (batch, n_kv_heads, 1, n_tokens, head_dim)
        .expand(batch_size, n_kv_head, n_repeat, n_tokens, head_dim)
        .flatten(1, 2)  # (batch, n_q_heads(n_kv_heads * n_repeat), n_tokens, head_dim)
    )


def _forward(self: Module, h: Hidden, *args, **kwargs) -> tuple[Hidden, None]:
    adapter: Adapter = self.adapter
    cache: Cache = self.cache
    n_q_heads, n_kv_heads, head_dim = adapter.dims(self)

    q, k, v = adapter.qkv(self, h)
    past_len = cache.len()
    cache.append(k, v)
    k_full, v_full = cache.get()  # 메모리병목: attention 계산 순간에는 FP복제본이 생기며 메모리 과다 사용

    # GQA의 경우 kv와 q의 head개수가 다를 수 있음
    k_full = _repeat_kv(k_full, n_q_heads // n_kv_heads)
    v_full = _repeat_kv(v_full, n_q_heads // n_kv_heads)

    q_len = q.shape[-2]
    kv_len = k_full.shape[-2]
    scale = adapter.scale(self)
    scores = torch.matmul(q, k_full.transpose(-1, -2)) * scale

    q_pos = torch.arange(q_len, device=q.device) + past_len  # 이번 query들의 위치
    k_pos = torch.arange(kv_len, device=q.device)  # key들의 위치
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # 각 q가 자기보다 미래의 kv를 보지 못하도록
    scores = scores.masked_fill(~causal, float("-inf"))

    heads = torch.matmul(torch.softmax(scores, dim=-1), v_full)

    attn_out = heads.transpose(1, 2).reshape(heads.shape[0], q_len, n_q_heads * head_dim)
    out = adapter.out(self, attn_out)
    return out, None


def patch(model: PreTrainedModel, adapter: Adapter) -> PreTrainedModel:
    for m in model.modules():
        if adapter.match(m):
            m.adapter = adapter
            m.cache = Cache()
            m.forward = types.MethodType(_forward, m)
    return model
