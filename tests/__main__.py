import runpy
from pathlib import Path

tests_dir = Path(__file__).parent
failed = 0
print()
for path in sorted(tests_dir.glob("test_*.py")):
    m = f"tests.{path.stem}"
    runpy.run_module(m, run_name="__main__")
    print()
