from abc import ABC, abstractmethod

from torch.nn import Module

from shapes import KV, Query, Hidden


class Adapter(ABC):
    def __init__(self):
        self.attn: type[Module] | None = None

    def match(self, m: Module) -> bool:
        return self.attn is not None and isinstance(m, self.attn)

    @abstractmethod
    def cache(self, **kwargs) -> str:
        pass

    @abstractmethod
    def layer_i(self, m: Module) -> int:
        pass

    @abstractmethod
    def dims(self, m: Module) -> tuple[
        int,  # n_q_heads
        int,  # n_kv_heads
        int,  # head_dim
    ]:
        pass

    @abstractmethod
    def qkv(self, m: Module, h: Hidden) -> tuple[Query, KV, KV]:
        pass

    @abstractmethod
    def scale(self, m: Module) -> float:
        *_, head_dim = self.dims(m)
        return head_dim**-0.5

    @abstractmethod
    def out(self, m: Module, h: Hidden) -> Hidden:
        pass

    @abstractmethod
    def rope(self, m: Module, q: Query, k: KV, **kwargs) -> tuple[Query, KV]:
        pass
