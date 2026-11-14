"""
beam_opt.session
================
The only place where the package launches MAPDL.

Every other module receives the session as an argument, so it can be
replaced by a test double.  Typical entry point:

>>> with MapdlSession(run_location="mapdl_tmp", nproc=4) as mapdl:
...     problem = OptimizationProblem(mapdl, model, ...)
...     result = optimize(problem, method="cobyla")

Design notes
------------
* A single MAPDL process is reused across all FEM evaluations (expensive to
  restart for each iteration).
* The context manager closes the process on normal exit, on exceptions and
  on Ctrl-C; an ``atexit`` hook covers sessions opened with ``start()`` and
  never closed, preventing orphaned MAPDL processes that hold the RST lock.
"""

from __future__ import annotations

import atexit
import logging
import os

from ansys.mapdl.core import launch_mapdl

logger = logging.getLogger(__name__)


class MapdlSession:
    """
    Launches and owns one PyMAPDL session.

    Parameters
    ----------
    run_location : str | None
        Working directory for MAPDL files.  Defaults to ``./mapdl_tmp``.
    nproc : int
        Number of MAPDL processors.
    jobname : str
        MAPDL job name; must match ``SolverSettings.jobname``.
    **launch_kwargs
        Extra arguments for ``ansys.mapdl.core.launch_mapdl``.
    """

    def __init__(
        self,
        run_location: str | None = None,
        nproc: int = 4,
        jobname: str = "beam_opt",
        **launch_kwargs,
    ) -> None:
        self.run_location = run_location or os.path.join(os.getcwd(), "mapdl_tmp")
        self.nproc = int(nproc)
        self.jobname = jobname
        self.launch_kwargs = launch_kwargs
        self.mapdl = None

    def start(self):
        """Launch MAPDL (once) and return the session."""
        if self.mapdl is None:
            os.makedirs(self.run_location, exist_ok=True)
            logger.info("Launching MAPDL in %s (nproc=%d)", self.run_location, self.nproc)
            self.mapdl = launch_mapdl(
                run_location=self.run_location,
                jobname=self.jobname,
                nproc=self.nproc,
                override=True,
                **self.launch_kwargs,
            )
            self.mapdl.filname(self.jobname)
            atexit.register(self.close)
        return self.mapdl

    def close(self) -> None:
        """Terminate the MAPDL process.  Safe to call multiple times."""
        if self.mapdl is None:
            return
        logger.info("Shutting down MAPDL session.")
        try:
            self.mapdl.exit()
        except Exception as exc:  # noqa: BLE001  best-effort teardown  # pragma: no cover
            logger.warning("MAPDL exit() raised: %s", exc)
        finally:
            self.mapdl = None
            atexit.unregister(self.close)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()
