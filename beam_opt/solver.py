"""
beam_opt.solver
===============
Static FEM solution with a mandatory convergence check.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ansys.mapdl.core.errors import MapdlRuntimeError

from .exceptions import SolutionNotConvergedError

logger = logging.getLogger(__name__)


@dataclass
class SolverSettings:
    """
    MAPDL solution knobs.

    Parameters
    ----------
    jobname : str
        MAPDL job name (``/FILNAME``); the result file is ``<jobname>.rst``.
    nsubst_target, nsubst_max, nsubst_min : int
        ``NSUBST`` arguments.
    nlgeom : bool
        Large-displacement analysis.
    """

    jobname: str = "beam_opt"
    nsubst_target: int = 10
    nsubst_max: int = 100
    nsubst_min: int = 5
    nlgeom: bool = True


def check_convergence(mapdl, expected_time: float = 1.0, atol: float = 1e-6) -> float:
    """
    Verify that the last solve converged up to the end of the load step.

    Must be called right after ``SOLVE``, while ``mapdl`` is still in
    ``/SOLU``; the session is left at ``/BEGIN`` level.

    Parameters
    ----------
    mapdl
        Active MAPDL session (or a test double).
    expected_time : float
        Load-step end time that a converged solution must reach [-].
    atol : float
        Absolute tolerance on the final time [-].

    Returns
    -------
    float
        Time of the last results set [-].

    Raises
    ------
    SolutionNotConvergedError
        If the convergence indicator is not 1 or the last results set
        stops before ``expected_time``.
    """
    cnvg = mapdl.get_value("ACTIVE", 0, "SOLU", "CNVG")
    mapdl.finish()

    mapdl.post1()
    mapdl.set("LAST")
    time = float(mapdl.get_value("ACTIVE", 0, "SET", "TIME"))
    mapdl.finish()

    if round(cnvg) != 1 or abs(time - expected_time) > atol:
        raise SolutionNotConvergedError(
            f"FEM solution not converged: CNVG={cnvg:g}, "
            f"final time {time:g} (expected {expected_time:g})"
        )
    return time


def solve(mapdl, settings: SolverSettings) -> None:
    """
    Run a static analysis of the model in *mapdl* and verify that it converged.

    Parameters
    ----------
    mapdl
        MAPDL session with the model already built (at ``/BEGIN`` level).
    settings : SolverSettings
        Solution knobs.

    Raises
    ------
    SolutionNotConvergedError
        If MAPDL does not produce an RST file, or the solution stops
        before the end of the load step (see ``check_convergence``).
    """
    rst_path = os.path.join(mapdl.directory, f"{settings.jobname}.rst")

    # Remove stale RST so we can detect a fresh successful solve
    if os.path.exists(rst_path):
        try:
            os.remove(rst_path)
        except PermissionError:
            logger.warning("Could not remove stale RST file: %s", rst_path)

    mapdl.run("/SOLU")
    mapdl.antype("STATIC")
    mapdl.nlgeom("ON" if settings.nlgeom else "OFF")
    mapdl.nsubst(settings.nsubst_target, settings.nsubst_max, settings.nsubst_min)
    mapdl.autots("ON")
    mapdl.outres("ALL", "ALL")
    try:
        mapdl.solve()
    except MapdlRuntimeError as err:
        if "converge" in str(err).lower():
            raise SolutionNotConvergedError(str(err)) from err
        raise

    # With NLGEOM an RST is written even for a partial solution:
    # its existence is necessary but not sufficient.
    if not os.path.exists(rst_path):
        mapdl.finish()
        raise SolutionNotConvergedError(f"FEM solution failed – RST not found at {rst_path}")

    check_convergence(mapdl)
