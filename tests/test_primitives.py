import torch

from common.primitives import _pack, _unpack, quant, dequant
from tests.harness import Checker
from tests.diagnostics import mae

torch.manual_seed(0)

# setting
batch_size, n_heads, n_tokens, head_dim = 2, 4, 128, 64
group_size = 32
x = torch.randn(batch_size, n_heads, n_tokens, head_dim)


checker = Checker("common/primitives.py")


checker.section("pack/unpack 무손실")
for bits in (2, 4):
    codes = torch.randint(0, 2**bits, (batch_size, n_heads, n_tokens, head_dim), dtype=torch.uint8)
    packed = _pack(codes, bits)
    unpacked = _unpack(packed, bits)
    checker.check("codes 완전 복원", torch.equal(unpacked, codes), f"bits={bits}")

checker.section("quant/dequant 라운드트립")
for bits in (2, 4):
    k_deq = dequant(quant(x, bits, group_size, -2), bits, group_size, -2)
    mae_k = mae(x, k_deq)
    checker.check(f"k shape 보존", k_deq.shape == x.shape, f"bits={bits}")
    cond = mae_k < (0.6 if bits == 2 else 0.15)
    checker.check(f"k 라운드트립 오차", cond, f"bits={bits} mae={mae_k:.4f}")

    v_deq = dequant(quant(x, bits, group_size, -1), bits, group_size, -1)
    mae_v = mae(x, v_deq)
    cond = mae_v < (0.6 if bits == 2 else 0.15)
    checker.check(f"v 라운드트립 오차", cond, f"bits={bits} mae={mae_v:.4f}")
