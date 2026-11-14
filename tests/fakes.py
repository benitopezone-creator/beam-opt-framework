"""
Test doubles for PyMAPDL (no ANSYS required).

``FakeMapdl`` records every command and answers queries from tables;
``canonical_model`` turns a recorded build sequence into a numbering-free
description of the model, so that two builders can be compared even if they
number nodes, elements and sections differently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FakeMapdl:
    """
    Minimal stand-in for a PyMAPDL session.

    Parameters
    ----------
    smisc : dict[int, array-like] | None
        ``{smisc_id: values}`` with one value per element (element IDs 1..n).
        Missing items read as zero.
    scalars : dict[tuple[str, str, str], float] | None
        Answers to ``get_value(entity, _, item1, it1num)`` keyed by
        ``(entity, item1, it1num)``, e.g. ``("ACTIVE", "SOLU", "CNVG")`` or
        ``("FSUM", "ITEM", "FX")``.
    directory : str | None
        Working directory; when set, ``solve()`` writes an empty
        ``<jobname>.rst`` there (jobname from the last ``filname`` call).
    """

    def __init__(
        self,
        smisc: dict[int, np.ndarray] | None = None,
        scalars: dict[tuple[str, str, str], float] | None = None,
        directory: str | None = None,
    ) -> None:
        self.smisc = {k: np.asarray(v, dtype=float) for k, v in (smisc or {}).items()}
        self.scalars = dict(scalars or {})
        self.directory = directory
        self.jobname = "file"
        self.calls: list[tuple[str, tuple]] = []
        self._etab: dict[str, int] = {}

    def filname(self, name: str, *args) -> None:
        self.calls.append(("filname", (name, *args)))
        self.jobname = name

    def solve(self, *args) -> None:
        self.calls.append(("solve", args))
        if self.directory is not None:
            Path(self.directory, f"{self.jobname}.rst").touch()

    def etable(self, label: str, item: str, comp: int) -> None:
        self.calls.append(("etable", (label, item, comp)))
        self._etab[label] = int(comp)

    def _etab_values(self, label: str) -> np.ndarray | None:
        return self.smisc.get(self._etab[str(label)])

    def get_value(self, entity: str, entnum: int, item1: str, it1num: str | int = "") -> float:
        self.calls.append(("get_value", (entity, entnum, item1, it1num)))
        if entity == "ELEM" and item1 == "ETAB":
            values = self._etab_values(str(it1num))
            return 0.0 if values is None else float(values[int(entnum) - 1])
        return float(self.scalars[(str(entity), str(item1), str(it1num))])

    def get_array(self, entity: str, entnum: int | str = "", item1: str = "", it1num="", *a):
        self.calls.append(("get_array", (entity, entnum, item1, it1num)))
        if entity == "ELEM" and item1 == "ETAB":
            n = max((len(v) for v in self.smisc.values()), default=0)
            values = self._etab_values(str(it1num))
            return np.zeros(n) if values is None else values.copy()
        raise NotImplementedError(f"FakeMapdl.get_array{(entity, entnum, item1, it1num)}")

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        # Any other MAPDL command is accepted and recorded.
        def _record(*args, **kwargs):
            self.calls.append((name, args))

        return _record


class StaticCantileverMapdl(FakeMapdl):
    """
    Closed-form stand-in for MAPDL on a straight cantilever along global X.

    The structure is statically determinate, so the internal forces do not
    depend on the sections: they follow from the nodal forces (``FY``/``FZ``)
    recorded during the build (any section), and the BEAM188 ``SMISC``
    bending stresses from the ``SECDATA`` of rectangular elements.  The beam is assumed clamped at
    its lowest-x node and oriented as BEAM188 does by default (local y =
    global Y, local z = global Z).  It lets a full optimisation loop run
    without ANSYS; it does not check MAPDL itself.

    Parameters
    ----------
    directory : str
        Working directory where ``solve()`` writes the empty ``.rst``.
    """

    def __init__(self, directory: str) -> None:
        scalars = {("ACTIVE", "SOLU", "CNVG"): 1.0, ("ACTIVE", "SET", "TIME"): 1.0}
        super().__init__(scalars=scalars, directory=directory)
        self.n_solves = 0
        self._reset()

    def _reset(self) -> None:
        self.calls = []
        self._nodes: dict[int, float] = {}  # node -> x [mm]
        self._secs: dict[int, tuple[float, ...]] = {}  # secnum -> SECDATA [mm]
        self._elems: list[tuple[int, int, int]] = []  # (secnum, node I, node J)
        self._forces: list[tuple[int, str, float]] = []
        self._cur_sec = 0
        self._def_sec = 0

    def clear(self, *args) -> None:
        self._reset()

    def n(self, num, x, y=0.0, z=0.0) -> None:
        self._nodes[int(num)] = float(x)

    def sectype(self, secnum, *args) -> None:
        self._def_sec = int(secnum)

    def secdata(self, *data) -> None:
        self._secs[self._def_sec] = tuple(float(v) for v in data)

    def secnum(self, secnum) -> None:
        self._cur_sec = int(secnum)

    def e(self, i, j) -> None:
        self._elems.append((self._cur_sec, int(i), int(j)))

    def f(self, node, direction, value) -> None:
        self._forces.append((int(node), str(direction), float(value)))

    def _resultants(self, x: float, direction: str) -> tuple[float, float]:
        """Shear [N] and bending moment [N·mm] of the forces acting beyond *x*."""
        loads = [
            (self._nodes[n], v) for n, d, v in self._forces if d == direction and self._nodes[n] > x
        ]
        return sum(v for _, v in loads), sum(v * (xk - x) for xk, v in loads)

    def solve(self, *args) -> None:
        super().solve(*args)
        self.n_solves += 1
        eps = 1e-9 * (max(self._nodes.values()) - min(self._nodes.values()))
        rows = []
        for sec, ni, nj in self._elems:
            data = self._secs[sec]
            row = {}
            # (first SMISC id, cut just inside the element)
            for first, stress0, x in (
                (1, 31, self._nodes[ni] + eps),
                (14, 36, self._nodes[nj] - eps),
            ):
                vy, mz = self._resultants(x, "FY")
                vz, my = self._resultants(x, "FZ")
                # Internal forces FX, MY, MZ, TQ, SFZ, SFY: valid for any section
                row.update({first: 0.0, first + 1: my, first + 2: -mz, first + 3: 0.0})
                row.update({first + 4: vz, first + 5: vy})
                if len(data) == 2:  # rectangle: BEAM188 bending stresses as well
                    b, h = data
                    sy = 6.0 * mz / (h * b**2)  # SByT = −Mz·y_max/Izz [MPa]
                    sz = 6.0 * my / (b * h**2)  # SBzT = My·z_max/Iyy [MPa]
                    row.update({stress0 + 1: sy, stress0 + 2: -sy})
                    row.update({stress0 + 3: sz, stress0 + 4: -sz})
            rows.append(row)
        self.smisc = {k: np.array([r.get(k, 0.0) for r in rows]) for k in rows[0]}
        for d in ("FX", "FY", "FZ"):
            total = sum(v for _, dd, v in self._forces if dd == d)
            self.scalars[("FSUM", "ITEM", d)] = -total


# -----------------------------------------------------------------------
# Canonical model
# -----------------------------------------------------------------------


def _num(v) -> float:
    return float(f"{float(v):.10g}")


def _xyz(coords) -> tuple[float, ...]:
    return tuple(_num(c) for c in coords)


def canonical_model(calls: list[tuple[str, tuple]]) -> dict:
    """
    Numbering-free description of the model written by a build sequence.

    Nodes are identified by their coordinates, sections by their data,
    element types by name; every list is sorted so that the result does not
    depend on the command order either.

    Parameters
    ----------
    calls : list[tuple[str, tuple]]
        ``FakeMapdl.calls`` recorded during a build.

    Returns
    -------
    dict
        JSON-serialisable description (lists of lists).
    """
    nodes: dict[int, tuple] = {}
    etypes: dict[int, str] = {}
    sections: dict[int, list] = {}
    materials: list = []
    elements: list = []
    constraints: list = []
    selection: set[int] | None = None  # None = all nodes
    cur = {"type": None, "mat": None, "secnum": None}
    last_sec: int | None = None

    def sel_nodes() -> list[int]:
        return sorted(nodes) if selection is None else sorted(selection)

    for name, args in calls:
        if name == "n":
            nodes[int(args[0])] = _xyz(args[1:4])
        elif name == "et":
            etypes[int(args[0])] = str(args[1])
        elif name == "mp":
            materials.append([str(args[0]), int(args[1]), _num(args[2])])
        elif name == "sectype":
            last_sec = int(args[0])
            sections[last_sec] = [str(args[1]), str(args[2]), [], ""]
        elif name == "secdata":
            sections[last_sec][2] = [_num(a) for a in args]
        elif name == "secoffset":
            sections[last_sec][3] = str(args[0])
        elif name in cur:
            cur[name] = int(args[0])
        elif name == "e":
            sec = sections.get(cur["secnum"])
            elements.append(
                [etypes[cur["type"]], cur["mat"], sec, [list(nodes[int(n)]) for n in args]]
            )
        elif name == "nsel":
            nid = int(args[3])
            selection = {nid} if args[0] == "S" else (set(sel_nodes()) | {nid})
        elif name == "allsel":
            selection = None
        elif name == "dsym":
            for nid in sel_nodes():
                constraints.append(["DSYM", list(nodes[nid]), str(args[0]), str(args[1])])
        elif name == "d":
            constraints.append(["D", list(nodes[int(args[0])]), str(args[1]), _num(args[2])])
        elif name == "f":
            constraints.append(["F", list(nodes[int(args[0])]), str(args[1]), _num(args[2])])
        elif name == "cerig":
            constraints.append(
                ["CERIG", list(nodes[int(args[0])]), list(nodes[int(args[1])]), str(args[2])]
            )
        elif name == "fsum":
            constraints.append(["FSUM", sorted(list(nodes[n]) for n in sel_nodes())])

    return {
        "nodes": sorted(list(c) for c in nodes.values()),
        "materials": sorted(materials),
        "elements": sorted(elements, key=repr),
        "constraints": sorted(constraints, key=repr),
    }
