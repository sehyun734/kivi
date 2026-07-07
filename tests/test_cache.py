import torch

from kivi.cache import Cache
from tests.diagnostics import mae
from tests.harness import Checker

torch.manual_seed(0)

# setting
k_bits, v_bits = 2, 2
batch_size, n_heads, n_tokens, head_dim = 1, 2, 200, 64
group_size, res_len = 32, 32
k = torch.randn(batch_size, n_heads, n_tokens, head_dim)
v = torch.randn(batch_size, n_heads, n_tokens, head_dim)


checker = Checker("kivi/cache.py")


checker.section("개별 토큰 append vs 한 번에 append")
cache_bulk = Cache(k_bits, v_bits, group_size, res_len)
cache_bulk.append(k, v)
k_bulk, v_bulk = cache_bulk.get()
cache_incr = Cache(k_bits, v_bits, group_size, res_len)
for i in range(n_tokens):
    cache_incr.append(k[..., i : i + 1, :], v[..., i : i + 1, :])
k_incr, v_incr = cache_incr.get()

checker.check("cache get k shape 보존", k_bulk.shape == k.shape, f"shape={str(tuple(k_bulk.shape))}")
checker.check("cache get v shape 보존", v_bulk.shape == v.shape, f"shape={str(tuple(v_bulk.shape))}")
checker.check("k_bulk == k_incr", torch.allclose(k_bulk, k_incr, atol=1e-5))
checker.check("v_bulk == v_incr", torch.allclose(v_bulk, v_incr, atol=1e-5))

checker.section("K_res(tail) 파트 FP 유지 여부")
n_res = cache_bulk.K_res.shape[2]
n_grp = cache_bulk.K_grp.codes.shape[-2]
checker.check("K_res 원본과 무손실", torch.equal(k_bulk[..., -n_res:, :], k[..., -n_res:, :]))
checker.check("V_res 원본과 무손실", torch.equal(v_bulk[..., -n_res:, :], v[..., -n_res:, :]))
mae_grp = mae(k_bulk[..., :n_grp, :], k[..., :n_grp, :])
checker.check("K_grp 파트 손실 존재", mae_grp > 1e-3, f"mae={mae_grp:.4f} n_grp={n_grp}")
