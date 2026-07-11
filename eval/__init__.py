import importlib
import pkgutil

# register 자동 등록을 위한 init import
for m in pkgutil.iter_modules(__path__):
    if m.name not in ("base",) and not m.name.startswith("__"):
        importlib.import_module(f"{__name__}.{m.name}")
