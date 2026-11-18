"""
beam_opt.optimization.history
=============================
CSV log of the FEM evaluations and restart from it.

One row per FEM evaluation, written as soon as it is available, so an
interrupted run loses nothing:

    eval, mass_g, objective, sigma_max, reaction, max_violation, failed, <variables…>

>>> problem = OptimizationProblem(..., history_file="run.csv")
>>> optimize(problem, "cobyla")                      # interrupted or finished
>>> x0 = best_design("run.csv")                      # later: restart from the best design
>>> optimize(problem, "slsqp", x0=x0)
"""

from __future__ import annotations

import csv
import os

import numpy as np

_FIXED = ["eval", "mass_g", "objective", "sigma_max", "reaction", "max_violation", "failed"]


class HistoryWriter:
    """
    Appends one CSV row per FEM evaluation.

    Parameters
    ----------
    path : str
        CSV file.  Created with a header if missing; otherwise rows are
        appended and the evaluation counter continues.
    variable_names : list[str]
        Column names of the design variables.

    Raises
    ------
    ValueError
        If an existing file has different columns (another problem).
    """

    def __init__(self, path: str, variable_names: list[str]) -> None:
        self.path = str(path)
        header = _FIXED + list(variable_names)
        self._count = 0
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if rows[0] != header:
                raise ValueError(f"{self.path}: columns do not match this problem")
            self._count = len(rows) - 1
        else:
            folder = os.path.dirname(self.path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def write(self, ev, max_violation: float, objective: float = float("nan")) -> None:
        """Append evaluation *ev* (an ``Evaluation``), its largest violation [-] and objective."""
        from .problem import TON_TO_G

        self._count += 1
        row = [
            self._count,
            repr(ev.mass * TON_TO_G),
            repr(float(objective)),
            repr(float(np.max(ev.sigma))),
            repr(float(ev.reaction)),
            repr(float(max_violation)),
            int(ev.failed),
            *(repr(float(v)) for v in ev.x),
        ]
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


def read_history(path: str) -> dict[str, np.ndarray]:
    """
    Read a history file.

    Returns
    -------
    dict[str, ndarray]
        One array per fixed column (``eval``, ``mass_g`` [g], ``objective``,
        ``sigma_max`` [MPa], ``reaction`` [N], ``max_violation`` [-],
        ``failed``) plus
        ``"x"``, shape (n_eval, n_vars) [mm], and ``"names"``, the variable
        names.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    data = np.array(body, dtype=float).reshape(len(body), len(header))
    out = {name: data[:, k] for k, name in enumerate(_FIXED)}
    out["x"] = data[:, len(_FIXED) :]
    out["names"] = np.array(header[len(_FIXED) :])
    return out


def best_design(path: str, feas_tol: float = 1e-4) -> np.ndarray:
    """
    Best feasible design of a history file (lowest objective), to restart an optimisation.

    Parameters
    ----------
    path : str
        History CSV.
    feas_tol : float
        Largest accepted constraint violation [-].

    Returns
    -------
    ndarray
        Design vector [mm].  If no design is feasible, the least violating
        one that did not fail.

    Raises
    ------
    ValueError
        If the file contains no successful evaluation.
    """
    h = read_history(path)
    ok = h["failed"] == 0
    if not np.any(ok):
        raise ValueError(f"{path}: no successful FEM evaluation")
    feasible = ok & (h["max_violation"] <= feas_tol)
    if np.any(feasible):
        idx = np.flatnonzero(feasible)
        return h["x"][idx[np.argmin(h["objective"][idx])]].copy()
    idx = np.flatnonzero(ok)
    return h["x"][idx[np.argmin(h["max_violation"][idx])]].copy()
