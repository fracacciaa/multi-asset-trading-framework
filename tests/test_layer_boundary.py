"""
Enforce the layer boundary: nothing above */production/ may import into it.

The boundary rule is that blend, core, and cross-asset layers must run from
committed parquets with no raw data present.  Production imports drag in
Bloomberg data-loading code and break that guarantee.

Allowed exceptions (explicitly named below):
  - rates/blend.py:       run_rates_full() rebuild path → rates.production.run
  - commodities/blend.py: _signal_panels() / build_signal_parquets() rebuild path
                           → commodities.production.*

Both are function-scoped (not at module level) so the import is only triggered
when the caller explicitly rebuilds from raw data.  All other production imports
in upper-layer files are bugs.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent

UPPER = [
    "rates/blend.py",
    "fx/blend.py",
    "commodities/blend.py",
    "equities/blend.py",
    "core/asset_blend.py",
    "core/signal_metrics.py",
    "core/main.py",
]

# Files allowed to import from their own sleeve's production layer.
# Key: relative path.  Value: set of production module prefixes permitted.
ALLOWLIST: dict[str, set[str]] = {
    "rates/blend.py":       {"rates.production"},
    "commodities/blend.py": {"commodities.production"},
}


def _imported_modules(rel: str) -> list[str]:
    src = (REPO_ROOT / rel).read_text()
    tree = ast.parse(src)
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name)
    return mods


@pytest.mark.parametrize("rel", UPPER)
def test_upper_layer_does_not_import_production(rel: str) -> None:
    allowed = ALLOWLIST.get(rel, set())
    bad = [
        m for m in _imported_modules(rel)
        if ".production" in m and not any(m.startswith(a) for a in allowed)
    ]
    assert not bad, (
        f"{rel} imports from production layer: {bad}\n"
        "Upper-layer files must read only from */signals/ parquets."
    )
