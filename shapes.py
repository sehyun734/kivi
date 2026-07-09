from jaxtyping import Float, UInt8
from torch import Tensor

Hidden = Float[Tensor, "batch_size n_tokens embed_dim"]

Query = Float[Tensor, "batch_size n_q_heads n_tokens head_dim"]
KV = Float[Tensor, "batch_size n_kv_heads n_tokens head_dim"]

Unpacked = UInt8[Tensor, "batch_size n_kv_heads n_tokens head_dim"]
Packed = UInt8[Tensor, "batch_size n_kv_heads n_tokens packed_head_dim"]

# k(channel-wise): (batch, n_kv_heads, n_grp,    1,     head_dim)
# v(token-wise):   (batch, n_kv_heads, n_tokens, n_grp, 1       )
Param = Float[Tensor, "batch_size n_kv_heads ..."]
