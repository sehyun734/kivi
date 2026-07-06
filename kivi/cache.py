from typing import Optional

import torch
from torch import Tensor
from jaxtyping import Float

from common.primitives import quant, dequant, Quantized
from common.shapes import KV


def _cat(old: Optional[Quantized], new: Quantized) -> Quantized:
    if old is None:
        return new
    return Quantized(
        torch.cat([old.codes, new.codes], dim=2),
        torch.cat([old.s, new.s], dim=2),
        torch.cat([old.z, new.z], dim=2),
    )


class Cache:
    def __init__(self, k_bits: int, v_bits: int, group_size: int, res_len: int):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.group_size = group_size
        self.res_len = res_len
        self.K_grp: Optional[Quantized] = None
        self.V_grp: Optional[Quantized] = None
        self.K_res: Optional[KV] = None
        self.V_res: Optional[KV] = None

    def len(self) -> int:
        n_grp = 0 if self.K_grp is None else self.K_grp.codes.shape[-2]
        n_res = 0 if self.K_res is None else self.K_res.shape[-2]
        return n_grp + n_res

    def append(self, k: KV, v: KV) -> None:
        self.K_res = k if self.K_res is None else torch.cat([self.K_res, k], dim=-2)
        self.V_res = v if self.V_res is None else torch.cat([self.V_res, v], dim=-2)

        n_res = self.K_res.shape[-2]
        n_flush = ((n_res - self.res_len) // self.group_size) * self.group_size
        if n_flush <= 0:
            return

        K_flush, self.K_res = self.K_res[..., :n_flush, :], self.K_res[..., n_flush:, :]
        V_flush, self.V_res = self.V_res[..., :n_flush, :], self.V_res[..., n_flush:, :]
        K_flush_q = quant(K_flush, self.k_bits, self.group_size, -2)
        V_flush_q = quant(V_flush, self.v_bits, self.group_size, -1)
        self.K_grp = _cat(self.K_grp, K_flush_q)
        self.V_grp = _cat(self.V_grp, V_flush_q)

    def get(self) -> tuple[
        Float[Tensor, "batch_size n_kv_heads n_tokens head_dim"],  # k
        Float[Tensor, "batch_size n_kv_heads n_tokens head_dim"],  # v
    ]:
        if self.K_grp is None:
            return self.K_res, self.V_res
        K_grp_deq = dequant(self.K_grp, self.k_bits, self.group_size, -2)
        V_grp_deq = dequant(self.V_grp, self.v_bits, self.group_size, -1)
        return (
            torch.cat([K_grp_deq, self.K_res], dim=-2),
            torch.cat([V_grp_deq, self.V_res], dim=-2),
        )
