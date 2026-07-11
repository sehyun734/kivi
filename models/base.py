from abc import ABC, abstractmethod

from torch.nn import Module

from shapes import KV, Query, Hidden

_ADAPTERS: dict[str, type["Adapter"]] = {}


def register(key: str):
    def deco(cls: type["Adapter"]) -> type["Adapter"]:  # 인스턴스가 아닌 클래스가 담기기에 type() 명시
        _ADAPTERS[key] = cls
        return cls

    return deco


def pick(model_name: str) -> "Adapter":
    for key in sorted(_ADAPTERS, key=len, reverse=True):
        if model_name.startswith(key):
            return _ADAPTERS[key]()
    raise KeyError


class Adapter(ABC):
    def __init__(self):
        self.attn: type[Module] | None = None  # 구현체에서 해당 attn을 갈아끼움

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
