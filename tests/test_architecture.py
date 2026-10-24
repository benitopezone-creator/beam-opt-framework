"""
Architectural rules of CLAUDE.md checked on the package source.

* Rule 4: MAPDL is launched only by ``beam_opt.session``.
* Rule 3: no application data in the package: no module-level float
  constants (material data, loads, bounds ...) and no ``Material(...)``
  instances at module level.
* Rule 6: no ``get_value`` on elements inside the package.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "beam_opt"
SOURCES = sorted(PACKAGE.rglob("*.py"))
IDS = [str(p.relative_to(PACKAGE)) for p in SOURCES]
ALLOWED_FLOAT_CONSTANTS = {"TON_TO_G"}  # unit conversion, not application data


@pytest.mark.parametrize("path", SOURCES, ids=IDS)
def test_only_session_launches_mapdl(path):
    if path.name == "session.py":
        return
    assert "launch_mapdl" not in path.read_text(encoding="utf-8")


def _module_level_assignments(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            yield from ((t, node.value) for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            yield node.target, node.value


@pytest.mark.parametrize("path", SOURCES, ids=IDS)
def test_no_application_data_at_module_level(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for target, value in _module_level_assignments(tree):
        if isinstance(value, ast.Constant) and isinstance(value.value, float):
            assert target.id in ALLOWED_FLOAT_CONSTANTS, f"float constant {target.id}"
        if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "Material":
            pytest.fail(f"module-level Material instance {target.id}")


@pytest.mark.parametrize("path", SOURCES, ids=IDS)
def test_no_per_element_get_value(path):
    assert 'get_value("ELEM"' not in path.read_text(encoding="utf-8")


def test_public_api_imports():
    import beam_opt

    for name in beam_opt.__all__:
        assert hasattr(beam_opt, name)
