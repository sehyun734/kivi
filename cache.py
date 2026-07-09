from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from quant import quant, dequant, Quantized
from shapes import KV


def _cat(old: Optional[Quantized], new: Quantized) -> Quantized:
    if old is None:
        return new
    return Quantized(
        torch.cat([old.codes, new.codes], dim=2),
        torch.cat([old.s, new.s], dim=2),
        torch.cat([old.z, new.z], dim=2),
    )


class Cache(DynamicCache):

    def __init__(self, k_bits: int, v_bits: int, grp_size: int, res_len: int):
        super().__init__()
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.grp_size = grp_size
        self.res_len = res_len
        # 레이어별 상태를 layer_i로 구분해 저장 (단일 객체가 모든 레이어의 KV 캐시 관리)
        self.K_grp: dict[int, Quantized] = {}  # Uint8로 저장
        self.V_grp: dict[int, Quantized] = {}
        self.K_res: dict[int, KV] = {}  # FP로 저장
        self.V_res: dict[int, KV] = {}

    # 인터페이스: masked multi-head를 위한 과거 토큰 길이
    def get_seq_length(self, layer_i: int = 0) -> int:
        n_grp = self.K_grp[layer_i].codes.shape[-2] if layer_i in self.K_grp else 0
        n_res = self.K_res[layer_i].shape[-2] if layer_i in self.K_res else 0
        return n_grp + n_res

    # 인터페이스: 새롭게 들어온 k,v를 저장하고 이 레이어의 전체(현재까지의) K,V를 반환
    def update(self, k: KV, v: KV, layer_i: int, *args, **kwargs) -> tuple[KV, KV]:
        self._append(k, v, layer_i)
        return self._get(layer_i)

    def _append(self, k: KV, v: KV, layer_i: int) -> None:
        K_res = k if layer_i not in self.K_res else torch.cat([self.K_res[layer_i], k], dim=-2)
        V_res = v if layer_i not in self.V_res else torch.cat([self.V_res[layer_i], v], dim=-2)

        n_res = K_res.shape[2]
        n_flush = ((n_res - self.res_len) // self.grp_size) * self.grp_size
        if n_flush > 0:  # residual이 res_len을 넘어서면 그룹 양자화하여 저장
            K_flush, K_res = K_res[..., :n_flush, :], K_res[..., n_flush:, :]
            V_flush, V_res = V_res[..., :n_flush, :], V_res[..., n_flush:, :]
            K_flush_q = quant(K_flush, self.k_bits, self.grp_size, -2)  # channel-wise
            V_flush_q = quant(V_flush, self.v_bits, self.grp_size, -1)  # token-wise
            self.K_grp[layer_i] = _cat(self.K_grp.get(layer_i), K_flush_q)
            self.V_grp[layer_i] = _cat(self.V_grp.get(layer_i), V_flush_q)
        self.K_res[layer_i], self.V_res[layer_i] = K_res, V_res

    def _get(self, layer_i: int) -> tuple[KV, KV]:
        if layer_i not in self.K_grp:
            return self.K_res[layer_i], self.V_res[layer_i]
        K_grp_deq = dequant(self.K_grp[layer_i], self.k_bits, self.grp_size, -2)
        V_grp_deq = dequant(self.V_grp[layer_i], self.v_bits, self.grp_size, -1)
        return (
            torch.cat([K_grp_deq, self.K_res[layer_i]], dim=-2),
            torch.cat([V_grp_deq, self.V_res[layer_i]], dim=-2),
        )
