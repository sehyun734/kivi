import importlib
import pkgutil

for m in pkgutil.iter_modules(__path__):
    if m.name not in ("base",) and not m.name.startswith("__"):
        importlib.import_module(f"{__name__}.{m.name}")
