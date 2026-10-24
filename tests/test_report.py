"""Smoke tests for the textual report and the plots."""

from __future__ import annotations

import numpy as np
import scenario as sc
from scipy.optimize import OptimizeResult
from test_phase1_regression import curved_frame_model, fake_mapdl, make_problem

from beam_opt.report import plot_results, print_report


def test_print_report(tmp_path, capsys):
    problem = make_problem(curved_frame_model(), fake_mapdl(str(tmp_path)))
    result = OptimizeResult(x=np.asarray(sc.X_DESIGN), success=True, nfev=1)
    print_report(result, problem)
    out = capsys.readouterr().out
    assert "OPTIMISATION REPORT" in out
    assert "Critical element" in out
    assert out.count("\n") > 20


def test_plot_results_saves_pngs(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    problem = make_problem(curved_frame_model(), fake_mapdl(str(tmp_path)))
    plot_results(problem, np.asarray(sc.X_DESIGN), save_path=str(tmp_path / "r"), show=False)
    assert (tmp_path / "r_sections.png").stat().st_size > 0
    assert (tmp_path / "r_stress.png").stat().st_size > 0
