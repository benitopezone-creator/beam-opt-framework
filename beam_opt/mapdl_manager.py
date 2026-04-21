"""
beam_opt.mapdl_manager
======================
MAPDL Singleton with a guaranteed ``exit()`` on interpreter shutdown.

Design notes
------------
* A single MAPDL process is reused across all FEM evaluations (expensive to
  restart for each iteration).
* ``atexit`` ensures the batch process is terminated even when the optimiser
  raises an exception or the user interrupts with Ctrl-C – preventing orphaned
  MAPDL processes that hold the RST lock.
* ``close()`` can also be called explicitly (e.g. at the end of ``main``).
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Optional

from ansys.mapdl.core import launch_mapdl, Mapdl

from .config import SolverSettings, DEFAULT_SOLVER

logger = logging.getLogger(__name__)


class MapdlManager:
    """
    Singleton wrapper around a PyMAPDL session.

    Usage
    -----
    >>> m = MapdlManager.get()          # first call launches MAPDL
    >>> m2 = MapdlManager.get()         # returns the same instance
    >>> MapdlManager.close()            # explicit teardown (optional)
    """

    _instance: Optional["MapdlManager"] = None
    _mapdl: Optional[Mapdl] = None

    def __new__(cls) -> "MapdlManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get(
        cls,
        run_location: Optional[str] = None,
        settings: SolverSettings = DEFAULT_SOLVER,
    ) -> Mapdl:
        """
        Return the active MAPDL instance, launching it on the first call.

        Parameters
        ----------
        run_location:
            Working directory for MAPDL output files.  Defaults to a
            ``mapdl_tmp`` sub-directory in the current working directory.
        settings:
            Solver settings dataclass; only ``nproc`` and ``jobname`` are
            used at launch time.
        """
        if cls._mapdl is None:
            if run_location is None:
                run_location = os.path.join(os.getcwd(), "mapdl_tmp")
            os.makedirs(run_location, exist_ok=True)

            logger.info("Launching MAPDL in %s (nproc=%d)", run_location, settings.nproc)

            cls._mapdl = launch_mapdl(
                run_location=run_location,
                jobname=settings.jobname,
                nproc=settings.nproc,
                override=True,
            )

            try:
                cls._mapdl.cwd(run_location)
            except Exception:
                pass

            cls._mapdl.filname(settings.jobname)

            # Register shutdown hook so MAPDL exits even on crash / KeyboardInterrupt
            atexit.register(cls._atexit_handler)

        return cls._mapdl

    @classmethod
    def close(cls) -> None:
        """
        Gracefully terminate the MAPDL process and reset the singleton.

        Safe to call multiple times.
        """
        if cls._mapdl is not None:
            logger.info("Shutting down MAPDL session.")
            try:
                cls._mapdl.exit()
            except Exception as exc:  # pragma: no cover
                logger.warning("MAPDL exit() raised: %s", exc)
            finally:
                cls._mapdl = None

    @classmethod
    def _atexit_handler(cls) -> None:
        """Called automatically at interpreter exit."""
        cls.close()
