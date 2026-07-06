from typing import Literal, NamedTuple

import torch

from common.shapes import KV, Packed, Param, Unpacked


class Quantized(NamedTuple):
    codes: Packed
    s: Param
    z: Param


def quant(x: KV, bits: int, group_size: int, dim: Literal[-2, -1], eps: float = 1e-5) -> Quantized:
    # -2(channel-wise): ... n_groups group_size head_dim
    # -1(token-wise):   ... n_tokens n_groups   group_size
    axis = dim % x.ndim
    dim_size = x.shape[axis]
    grouped = x.reshape(*x.shape[:axis], dim_size // group_size, group_size, *x.shape[axis + 1 :])
    qmax: int = 2**bits - 1
    z = grouped.amin(dim=axis + 1, keepdim=True)
    s = (grouped.amax(dim=axis + 1, keepdim=True) - z).clamp(min=eps) / qmax
    codes = torch.round((grouped - z) / s).clamp(0, qmax).to(torch.uint8).reshape(x.shape)
    return Quantized(_pack(codes, bits), z.float(), s.float())


def dequant(q: Quantized, bits: int, group_size: int, dim: Literal[-2, -1]) -> KV:
    codes = _unpack(q.codes, bits)
    axis = dim % codes.ndim
    dim_size = codes.shape[axis]
    grouped = codes.reshape(
        *codes.shape[:axis], dim_size // group_size, group_size, *codes.shape[axis + 1 :]
    )
    x = grouped.float() * q.s + q.z
    return x.reshape(codes.shape)


def _pack(codes: Unpacked, bits: int) -> Packed:
    n_per_byte = 8 // bits
    *lead, last = codes.shape
    # 계산 시 overflow 방지 위한 int32
    fields = codes.reshape(*lead, last // n_per_byte, n_per_byte).to(torch.int32)
    shifts = torch.arange(n_per_byte, device=codes.device, dtype=torch.int32) * bits
    # 비트필드가 겹치지 않아 sum이 OR의 역할을 수행하여 한 바이트로 병합
    return (fields << shifts).sum(-1).to(torch.uint8)


def _unpack(codes: Packed, bits: int) -> Unpacked:
    n_per_byte = 8 // bits
    mask = (1 << bits) - 1
    shifts = torch.arange(n_per_byte, device=codes.device, dtype=torch.int32) * bits
    fields = (codes.to(torch.int32).unsqueeze(-1) >> shifts) & mask
    *lead, n_bytes, _ = fields.shape
    return fields.reshape(*lead, n_bytes * n_per_byte).to(torch.uint8)
