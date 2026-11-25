import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("MATF_DATA_DIR", "~/Downloads")).expanduser()


def raw(name: str) -> Path:
    """Resolve a raw Bloomberg export. Only production/ layers may call this."""
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"{name} not found in {DATA_DIR}.\n"
            f"Set MATF_DATA_DIR to the folder holding the Bloomberg exports, or run "
            f"from committed parquets: the blend, TC and cross-asset layers need no raw data."
        )
    return p
