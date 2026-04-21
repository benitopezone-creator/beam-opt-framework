"""
AI Helmet – optimisation entry point.

Usage
-----
    python run_optimization.py [options]

Run ``python run_optimization.py --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys

import numpy as np

# Allow running from examples/aihelmet without installing the package
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from beam_opt import BeamModel, BeamOptimizer, MapdlManager
from beam_opt.config import SolverSettings, UX_IMP, SIGMA_S, FS, F_REACT_MAX
from beam_opt.report import print_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dimensional optimisation of the AI Helmet beam structure."
    )
    p.add_argument(
        "--config", default="v05", choices=["v05", "v07"],
        help="Geometry configuration module (default: v05).",
    )
    p.add_argument(
        "--n-ctrl", type=int, default=8,
        help="Number of spline control points (default: 8).",
    )
    p.add_argument(
        "--nproc", type=int, default=4,
        help="Number of MAPDL parallel processes (default: 4).",
    )
    p.add_argument(
        "--run-location", default=None,
        help="Working directory for MAPDL files (default: ./mapdl_tmp).",
    )
    p.add_argument(
        "--save-plots", default=None, metavar="PREFIX",
        help="Path prefix for PNG plot output (e.g. 'results/v05').",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Suppress matplotlib windows.",
    )
    p.add_argument(
        "--maxiter", type=int, default=3000,
        help="Maximum COBYLA iterations (default: 3000).",
    )
    return p.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── Load geometry config ──────────────────────────────────────────
    cfg_module = importlib.import_module(f"config_{args.config}")
    nodes_coords   = cfg_module.NODES_COORDS
    additional_nodes = cfg_module.ADDITIONAL_NODES
    master_node    = cfg_module.MASTER_NODE
    fixed_elements = cfg_module.FIXED_ELEMENTS
    uz_imp         = getattr(cfg_module, "UZ_IMP", 90.0)

    # ── Solver settings ───────────────────────────────────────────────
    settings = SolverSettings(nproc=args.nproc)

    # ── Build model ───────────────────────────────────────────────────
    run_location = args.run_location or os.path.join(os.getcwd(), "mapdl_tmp")

    model = BeamModel(
        nodes_coords=nodes_coords,
        fixed_elements=fixed_elements,
        additional_nodes=additional_nodes,
        master_node=master_node,
        ux_imp=UX_IMP,
        uz_imp=uz_imp,
        settings=settings,
    )

    # Ensure MAPDL is launched from the desired location
    MapdlManager.get(run_location=run_location, settings=settings)

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = BeamOptimizer(
        model=model,
        n_ctrl=args.n_ctrl,
        sigma_s=SIGMA_S,
        fs=FS,
        f_react_max=F_REACT_MAX,
    )

    logger.info("Fixed-element mass: %.3f g", optimizer.mass_fixed * 1e6)
    logger.info("Starting optimisation …")

    # ── Run ───────────────────────────────────────────────────────────
    result = optimizer.run(maxiter=args.maxiter)
    print(result)

    # ── Report ────────────────────────────────────────────────────────
    print_report(
        result,
        optimizer,
        show_plots=not args.no_plots,
        save_plots_path=args.save_plots,
    )

    # ── Explicit MAPDL shutdown ───────────────────────────────────────
    # atexit handles crashes, but calling close() here gives a clean exit
    # message in the log and prevents any stale lock files.
    MapdlManager.close()


if __name__ == "__main__":
    main()
