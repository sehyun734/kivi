from typing import Literal

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from quant import Quantized, _pack
from shapes import KV


@triton.jit
def _quant_per_token(
    x_ptr,  # 양자화할 원본 FP16 (head_dim, )
    codes_ptr,
    s_ptr,
    z_ptr,
    head_dim,
    grp_size,
    n_grp,
    q_max,
    eps,
    grp_size_padded,  # GPU는 2의 거듭제곱 단위로 처리하므로 grp_size를 올림한 값
):
    pid = tl.program_id(0)
    row_i = pid // n_grp
    grp_i = pid % n_grp

    offsets = tl.arange(0, grp_size_padded)  # 그룹 내에서의 인덱스 리스트
    mask = offsets < grp_size  # 실제 grp_size 부분만 로드하기 위한 마스크
    grp_start = row_i * head_dim + grp_i * grp_size  # 펴진 1차원 메모리 기준 그룹 시작 위치
    x = tl.load(x_ptr + grp_start + offsets, mask=mask, other=0.0)  # (grp_size_padded,) 그룹 밖은 0.0

    inf = 3.0e38
    x_max = tl.max(tl.where(mask, x, -inf), axis=0)  # 패딩 칸을 -inf로 채워 max에서 제외
    x_min = tl.min(tl.where(mask, x, inf), axis=0)
    s = tl.maximum((x_max - x_min) / q_max, eps)
    x_q = tl.extra.libdevice.round((x - x_min) / s)
    x_q = tl.minimum(tl.maximum(x_q, 0.0), q_max).to(tl.uint8)

    tl.store(codes_ptr + grp_start + offsets, x_q, mask=mask)
    tl.store(z_ptr + pid, x_min)
    tl.store(s_ptr + pid, s)


@triton.jit
def _quant_per_channel(
    x_ptr,  # 열 단위라 matrix를 작업 단위로 처리 (n_tokens, head_dim)
    codes_ptr,
    s_ptr,
    z_ptr,
    n_tokens,
    head_dim,
    grp_size,
    n_grp,
    q_max,
    eps,
    grp_size_padded,
    head_dim_padded,  # 작업 단위가 2D이므로 세로축(head_dim)도 2의 거듭제곱으로 올림
):
    pid = tl.program_id(0)
    matrix_i = pid // n_grp
    grp_i = pid % n_grp

    row_offsets = tl.arange(0, grp_size_padded)
    col_offsets = tl.arange(0, head_dim_padded)
    row_mask = row_offsets < n_tokens
    col_mask = col_offsets < head_dim
    mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)

    # 담당 matrix의 시작 위치 + matrix 내 해당 그룹의 행 오프셋
    matrix_start = matrix_i * n_tokens * head_dim + grp_i * grp_size * head_dim
    offsets = matrix_start + row_offsets.unsqueeze(1) * head_dim + col_offsets.unsqueeze(0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # (grp_size_padded, head_dim_padded)

    inf = 3.0e38
    x_max = tl.max(tl.where(mask, x, -inf), axis=0)  # (head_dim_padded, )
    x_min = tl.min(tl.where(mask, x, inf), axis=0)
    s = tl.maximum((x_max - x_min) / q_max, eps)
    x_q = tl.extra.libdevice.round((x - x_min.unsqueeze(0)) / s.unsqueeze(0))
    x_q = tl.minimum(tl.maximum(x_q, 0.0), q_max).to(tl.uint8)

    tl.store(codes_ptr + offsets, x_q, mask=mask)
    param_offsets = matrix_i * n_grp * head_dim + grp_i * head_dim + col_offsets
    tl.store(z_ptr + param_offsets, x_min, mask=col_mask)
    tl.store(s_ptr + param_offsets, s, mask=col_mask)


def quant_triton(x: KV, bits: int, grp_size: int, dim: Literal[-2, -1], eps: float = 1e-5) -> Quantized:
    batch_size, n_heads, n_tokens, head_dim = x.shape
    axis = dim % x.ndim
    dim_size = x.shape[axis]
    n_grp = dim_size // grp_size
    q_max = 2**bits - 1

    if axis == 3:  # token-wise (v)
        n_rows = batch_size * n_heads * n_tokens
        x_rows = x.contiguous().reshape(n_rows, head_dim)

        codes = torch.empty((n_rows, head_dim), dtype=torch.uint8, device=x.device)
        s = torch.empty((n_rows * n_grp), dtype=torch.float32, device=x.device)
        z = torch.empty_like(s)

        _quant_per_token[(n_rows * n_grp,)](
            x_rows,
            codes,
            s,
            z,
            head_dim,
            grp_size,
            n_grp,
            q_max,
            eps,
            triton.next_power_of_2(grp_size),
        )
        packed = _pack(codes.reshape(batch_size, n_heads, n_tokens, head_dim), bits)
        return Quantized(
            packed,
            s.reshape(batch_size, n_heads, n_tokens, n_grp, 1),
            z.reshape(batch_size, n_heads, n_tokens, n_grp, 1),
        )
    elif axis == 2:  # channel-wise (k)
        n_matrices = batch_size * n_heads
        x_matrices = x.contiguous().reshape(n_matrices, n_tokens, head_dim)

        codes = torch.empty((n_matrices, n_tokens, head_dim), dtype=torch.uint8, device=x.device)
        s = torch.empty((n_matrices, n_grp * head_dim), dtype=torch.float32, device=x.device)
        z = torch.empty_like(s)

        _quant_per_channel[(n_matrices * n_grp,)](
            x_matrices,
            codes,
            s,
            z,
            n_tokens,
            head_dim,
            grp_size,
            n_grp,
            q_max,
            eps,
            triton.next_power_of_2(grp_size),
            triton.next_power_of_2(head_dim),
        )
        packed = _pack(codes.reshape(batch_size, n_heads, n_tokens, head_dim), bits)
        return Quantized(
            packed,
            s.reshape(batch_size, n_heads, n_grp, 1, head_dim),
            z.reshape(batch_size, n_heads, n_grp, 1, head_dim),
        )
    else:
        raise ValueError
