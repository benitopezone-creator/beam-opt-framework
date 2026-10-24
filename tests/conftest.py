"""Shared pytest fixtures (no ANSYS required except for ``mapdl_session``)."""

from __future__ import annotations

import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest


class ConstantLaw:
    """Section law returning constant ``b`` and ``h`` [mm] everywhere."""

    def __init__(self, b: float, h: float) -> None:
        self.b = float(b)
        self.h = float(h)

    def eval(self, s):
        s = np.asarray(s, dtype=float)
        return np.full_like(s, self.b), np.full_like(s, self.h)


@pytest.fixture(scope="session")
def mapdl_session(tmp_path_factory):
    """
    Real MAPDL session shared by all ``mapdl``-marked tests.

    Skips the requesting test when MAPDL cannot be launched (not installed or
    no license), so that a plain ``pytest`` run stays usable without ANSYS.
    """
    from ansys.mapdl.core import launch_mapdl
    from ansys.tools.common.path import get_mapdl_path

    # Detect the executable without prompting on stdin (the launcher would);
    # a remote instance can be targeted through PYMAPDL_IP instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "No path found for mapdl"
        local_exec = get_mapdl_path(allow_input=False)
    if not os.environ.get("PYMAPDL_IP") and not local_exec:
        pytest.skip("MAPDL not available: no local executable and PYMAPDL_IP unset")

    run_location = tmp_path_factory.mktemp("mapdl")
    mapdl = launch_mapdl(run_location=str(run_location), override=True)
    yield mapdl
    mapdl.exit()


@pytest.fixture
def straight_model():
    """Factory for a minimal model stub: ``n_elem`` straight elements of unit length."""

    def _make(n_elem: int = 1):
        s = np.arange(n_elem + 1, dtype=float)
        return SimpleNamespace(n_elem=n_elem, s=s, s_elem=0.5 * (s[:-1] + s[1:]))

    return _make
