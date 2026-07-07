class Checker:
    def __init__(self, title: str | None = None):
        if title:
            print(f"=== {title} ===")

    def section(self, title: str) -> None:
        print(f"{title}")

    def check(self, name: str, cond: bool, tag: str = "") -> bool:
        status = "  [PASS]" if cond else " *[FAIL]"
        print(f"{status} {name}{f' ({tag})' if tag else ""}")
