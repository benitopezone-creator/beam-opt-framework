"""Tests for the FEM solve and its convergence check."""

from __future__ import annotations

import pytest
from fakes import FakeMapdl

from beam_opt.exceptions import SolutionNotConvergedError
from beam_opt.solver import SolverSettings, check_convergence, solve


def _fake(cnvg: float, time: float, directory=None) -> FakeMapdl:
    return FakeMapdl(
        scalars={("ACTIVE", "SOLU", "CNVG"): cnvg, ("ACTIVE", "SET", "TIME"): time},
        directory=directory,
    )


def test_converged_solution_passes():
    assert check_convergence(_fake(cnvg=1, time=1.0)) == pytest.approx(1.0)


def test_partial_load_raises():
    # NLGEOM run that stopped at 40 % of the load: RST exists but is not valid.
    with pytest.raises(SolutionNotConvergedError, match="final time 0.4"):
        check_convergence(_fake(cnvg=1, time=0.4))


def test_convergence_flag_zero_raises():
    with pytest.raises(SolutionNotConvergedError, match="CNVG=0"):
        check_convergence(_fake(cnvg=0, time=1.0))


def test_is_a_runtime_error():
    # Existing callers that catch RuntimeError keep working.
    assert issubclass(SolutionNotConvergedError, RuntimeError)


def test_solve_checks_convergence_even_if_rst_exists(tmp_path):
    fake = _fake(cnvg=1, time=0.4, directory=str(tmp_path))
    fake.filname("job")
    with pytest.raises(SolutionNotConvergedError, match="final time 0.4"):
        solve(fake, SolverSettings(jobname="job"))
    assert (tmp_path / "job.rst").exists()


def test_solve_without_rst_raises(tmp_path):
    fake = _fake(cnvg=1, time=1.0, directory=str(tmp_path))
    fake.solve = lambda *a: None  # MAPDL wrote no result file
    with pytest.raises(SolutionNotConvergedError, match="RST not found"):
        solve(fake, SolverSettings(jobname="job"))


def test_solve_converged(tmp_path):
    fake = _fake(cnvg=1, time=1.0, directory=str(tmp_path))
    fake.filname("job")
    solve(fake, SolverSettings(jobname="job", nlgeom=False))
    assert ("nlgeom", ("OFF",)) in fake.calls


@pytest.mark.mapdl
def test_check_convergence_on_real_cantilever(mapdl_session):
    mapdl = mapdl_session
    mapdl.clear()
    mapdl.prep7()
    mapdl.et(1, "BEAM188")
    mapdl.mp("EX", 1, 2.0e5)
    mapdl.mp("PRXY", 1, 0.3)
    mapdl.sectype(1, "BEAM", "RECT")
    mapdl.secdata(2.0, 3.0)
    mapdl.n(1, 0, 0, 0)
    mapdl.n(2, 100, 0, 0)
    mapdl.e(1, 2)
    mapdl.d(1, "ALL", 0)
    mapdl.f(2, "FY", -1.0)
    mapdl.finish()

    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.solve()
    assert check_convergence(mapdl) == pytest.approx(1.0)
