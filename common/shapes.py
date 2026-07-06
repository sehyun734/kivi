from jaxtyping import Float, UInt8, Int
from torch import Tensor

Hidden = Float[Tensor, "batch_size n_tokens embed_dim"]

Query = Float[Tensor, "batch_size n_q_heads n_tokens head_dim"]
KV = Float[Tensor, "batch_size n_kv_heads n_tokens head_dim"]

Unpacked = UInt8[Tensor, "batch_size n_kv_heads n_tokens head_dim"]
Packed = UInt8[Tensor, "batch_size n_kv_heads n_tokens packed_head_dim"]

# k: (batch, n_kv_heads, n_groups, 1,        head_dim)
# v: (batch, n_kv_heads, n_tokens, n_groups, 1       )
Param = Float[Tensor, "batch n_kv_heads ..."]

Pos = Int[Tensor, "n_tokens"]
