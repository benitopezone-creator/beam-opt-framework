"""
beam_opt.apdl
=============
Import of APDL scripts and export of optimised designs, without ANSYS.

Import
------
``read_apdl`` interprets a script written with direct generation (nodes and
elements) and returns a ready ``StructuralModel``:

>>> imported = read_apdl("frame.inp", sigma_y=355.0)
>>> problem = imported.problem(NativeSolver(), n_ctrl=6, bounds={"b": (2, 8), "h": (2, 12)},
...                            safety_factor=1.5, f_react_max=200.0)
>>> result = optimize(problem, x0=imported.x0(problem))

The elements to optimise are those of the element component ``OPT_BEAM``
(``ESEL`` + ``CM,OPT_BEAM,ELEM`` in the script); they must form one chain.
Every other beam becomes a ``FixedBeam`` and every shell a ``FixedShell``.

Supported commands: parameters (``A = ...``, ``*SET``), expressions with the
usual functions, ``*DO``/``*ENDDO``, ``*AFUN``, ``ET``, ``MP`` (EX, PRXY,
NUXY, DENS), ``SECTYPE``/``SECDATA`` (beam subtypes RECT, CSOLID, CTUBE,
HREC, I, ASEC; shells), ``N``, ``E``, ``EN``, ``TYPE``, ``MAT``, ``SECNUM``,
``NSEL``, ``ESEL``, ``ALLSEL``, ``CM``, ``D``, ``F``, ``DSYM``, ``CERIG``.
Solution and post-processing commands are ignored and listed in
``ApdlImport.notes``.  Solid modelling (``K``, ``L``, ``LMESH``, ...),
``*IF`` and macros are not supported: mesh the model in ANSYS and export
nodes and elements, or write them directly.

Export
------
``write_apdl`` writes a design as an APDL input file, with one section per
optimisable element and the same ``OPT_BEAM`` component, so that it can be
verified in ANSYS or imported again.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .geometry import BeamMesh
from .materials import Material
from .model import (
    Clamp,
    FixedBeam,
    FixedShell,
    ImposedDisplacement,
    ModelBuilder,
    NodalForce,
    OptimizableBeam,
    RigidRegion,
    StructuralModel,
    Symmetry,
)
from .section_law import SectionLaw
from .sections import MAPDL_SUBTYPES, SectionShape

#: Name of the element component that marks the beam to optimise.
OPT_COMPONENT = "OPT_BEAM"

_DOFS = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")
_LOADS = ("FX", "FY", "FZ", "MX", "MY", "MZ")
_BEAMS = {"BEAM188", "188", "BEAM189", "189"}
_SHELLS = {"SHELL181", "181"}
# Commands that do not change the model: skipped, reported in the notes
_IGNORED = {
    "FINISH", "FINI", "SOLVE", "ANTYPE", "NLGEOM", "NSUBST", "AUTOTS", "OUTRES", "TIME",
    "SECOFFSET", "SET", "ETABLE", "FSUM", "PRNSOL", "PLNSOL", "PRRSOL", "EPLOT", "NPLOT",
    "SAVE", "RESUME", "KBC", "EQSLV", "CNVTOL", "NEQIT", "LSWRITE", "REAL", "R", "ALLS",
    "NUMMRG", "NUMCMP", "SHPP", "KEYOPT", "DELTIM", "PSTRES", "LNSRCH",
}  # fmt: skip


_SILENT = {"/PREP7", "/SOLU", "/POST1", "/COM", "/TITLE", "/CLEAR", "/FILNAME", "FINISH", "SOLVE"}


class ApdlError(ValueError):
    """An APDL script could not be interpreted; the message carries the line number."""


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _functions(degrees: bool) -> dict:
    to_rad = math.radians if degrees else (lambda v: v)
    from_rad = math.degrees if degrees else (lambda v: v)
    return {
        "SIN": lambda v: math.sin(to_rad(v)),
        "COS": lambda v: math.cos(to_rad(v)),
        "TAN": lambda v: math.tan(to_rad(v)),
        "ASIN": lambda v: from_rad(math.asin(v)),
        "ACOS": lambda v: from_rad(math.acos(v)),
        "ATAN": lambda v: from_rad(math.atan(v)),
        "ATAN2": lambda a, b: from_rad(math.atan2(a, b)),
        "SQRT": math.sqrt,
        "ABS": abs,
        "EXP": math.exp,
        "LOG": math.log,
        "LOG10": math.log10,
        "NINT": lambda v: float(round(v)),
        "MOD": math.fmod,
        "MIN": min,
        "MAX": max,
        "SIGN": lambda a, b: math.copysign(a, b),
    }


def _evaluate(expr: str, params: dict[str, float], degrees: bool) -> float:
    """Value of an APDL arithmetic expression (no arbitrary code is executed)."""
    funcs = _functions(degrees)

    def walk(node) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.Name):
            name = node.id.upper()
            if name == "PI":
                return math.pi
            if name not in params:
                raise KeyError(f"undefined parameter {node.id}")
            return params[name]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id.upper()
            if name in funcs:
                return float(funcs[name](*(walk(a) for a in node.args)))
        raise ValueError(f"unsupported expression {expr!r}")

    return walk(ast.parse(expr.strip(), mode="eval"))


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@dataclass
class ApdlImport:
    """
    Result of ``read_apdl``.

    Attributes
    ----------
    model : StructuralModel
        The structure; nodes are named ``n<number>`` after the script numbers
        (plus ``"start"``/``"end"`` for the ends of the optimisable beam).
    section_factory : callable
        Section factory of the optimisable beam (also ``model.beam.section_factory``).
    param_names : tuple[str, ...]
        Section parameters to give bounds for (e.g. ``("b", "h")``).
    initial : dict[str, ndarray]
        Section parameters of the script, one value per optimisable element.
    elements : list[int]
        Script numbers of the optimisable elements, in chain order.
    notes : list[str]
        Commands that were ignored, with their line numbers.
    """

    model: StructuralModel
    section_factory: object
    param_names: tuple[str, ...]
    initial: dict[str, np.ndarray]
    elements: list[int]
    notes: list[str] = field(default_factory=list)

    def initial_law(self) -> SectionLaw:
        """The sections of the script as a piecewise-linear law through the elements."""
        mesh = self.model.beam.mesh
        return SectionLaw(mesh.s_elem, self.initial, interpolator="linear")

    def problem(self, solver, *, n_ctrl: int, bounds, reaction_nodes=None, **kwargs):
        """
        ``OptimizationProblem`` for the imported model.

        ``reaction_nodes`` defaults to every node with a prescribed
        displacement; the other arguments are those of
        ``OptimizationProblem``.
        """
        from .optimization import OptimizationProblem

        if reaction_nodes is None:
            reaction_nodes = sorted(
                {
                    n
                    for bc in self.model.bcs
                    if isinstance(bc, Clamp | ImposedDisplacement | Symmetry)
                    for n in bc.node_names()
                }
            )
        return OptimizationProblem(
            solver,
            self.model,
            n_ctrl=n_ctrl,
            bounds=bounds,
            reaction_nodes=reaction_nodes,
            **kwargs,
        )

    def x0(self, problem) -> np.ndarray:
        """Design vector that reproduces the script sections at the control points."""
        law = self.initial_law().eval(problem.s_ctrl, clip=False)
        x = np.concatenate([law[name] for name in problem.bounds])
        return np.clip(x, problem.lower, problem.upper)


class _Reader:
    """State machine that interprets the statements of a script."""

    def __init__(self, parameters: dict[str, float] | None) -> None:
        self.params = {k.upper(): float(v) for k, v in (parameters or {}).items()}
        self.degrees = False
        self.etypes: dict[int, str] = {}
        self.materials: dict[int, dict[str, float]] = {}
        self.sections: dict[int, dict] = {}
        self.nodes: dict[int, np.ndarray] = {}
        self.elements: dict[int, dict] = {}
        self.cur = {"TYPE": 1, "MAT": 1, "SECNUM": 1}
        self.last_section = 0
        self.nsel: set[int] | None = None
        self.esel: set[int] | None = None
        self.components: dict[str, list[int]] = {}
        self.fixed_dofs: dict[int, dict[int, float]] = {}
        self.loads: list[tuple[int, str, float]] = []
        self.symmetry: list[tuple[int, str]] = []
        self.rigid: dict[int, list[int]] = {}
        self.notes: list[str] = []

    # -- helpers -------------------------------------------------------

    def num(self, text: str, default: float = 0.0) -> float:
        text = text.strip()
        return default if not text else _evaluate(text, self.params, self.degrees)

    def selected_nodes(self) -> list[int]:
        return sorted(self.nodes if self.nsel is None else self.nsel)

    def selected_elements(self) -> list[int]:
        return sorted(self.elements if self.esel is None else self.esel)

    def node_range(self, args: list[str], first: int, i_end: int, i_inc: int) -> list[int]:
        """Nodes addressed by a ``D``/``F`` command: one, a range, or the selection."""
        if args[first].strip().upper() == "ALL":
            return self.selected_nodes()
        start = int(self.num(args[first]))
        end = int(self.num(args[i_end])) if len(args) > i_end and args[i_end].strip() else start
        inc = int(self.num(args[i_inc])) if len(args) > i_inc and args[i_inc].strip() else 1
        return list(range(start, end + 1, inc))

    @staticmethod
    def _select(current, universe, mode: str, picked: set[int]):
        if mode in ("S", ""):
            return picked
        base = set(universe) if current is None else current
        if mode == "A":
            return base | picked
        if mode == "R":
            return base & picked
        if mode == "U":
            return base - picked
        raise ValueError(f"unsupported selection type {mode!r}")

    # -- commands ------------------------------------------------------

    def command(self, name: str, a: list[str]) -> None:
        a = a + [""] * 12
        up = [s.strip().upper() for s in a]

        if name == "ET":
            self.etypes[int(self.num(a[0]))] = up[1]
        elif name == "MP":
            self.materials.setdefault(int(self.num(a[1], 1)), {})[up[0]] = self.num(a[2])
        elif name == "SECTYPE":
            self.last_section = int(self.num(a[0]))
            self.sections[self.last_section] = {"kind": up[1], "subtype": up[2], "data": ()}
        elif name == "SECDATA":
            values = [s for s in a if s.strip()]
            self.sections[self.last_section]["data"] = tuple(self.num(s) for s in values)
        elif name in ("TYPE", "MAT", "SECNUM"):
            self.cur[name] = int(self.num(a[0]))
        elif name == "N":
            self.nodes[int(self.num(a[0]))] = np.array(
                [self.num(a[1]), self.num(a[2]), self.num(a[3])]
            )
        elif name in ("E", "EN"):
            fields = a[1:] if name == "EN" else a
            number = int(self.num(a[0])) if name == "EN" else max(self.elements, default=0) + 1
            nodes = [int(self.num(s)) for s in fields if s.strip()]
            self.elements[number] = {
                "type": self.cur["TYPE"],
                "mat": self.cur["MAT"],
                "sec": self.cur["SECNUM"],
                "nodes": nodes,
            }
        elif name == "NSEL":
            if up[0] in ("ALL", "NONE"):
                self.nsel = None if up[0] == "ALL" else set()
                return
            lo = self.num(a[3])
            hi = self.num(a[4], lo) if a[4].strip() else lo
            if up[1] in ("NODE", ""):
                inc = int(self.num(a[5], 1)) or 1
                picked = set(range(int(lo), int(hi) + 1, inc)) & set(self.nodes)
            elif up[1] == "LOC" and up[2] in ("X", "Y", "Z"):
                axis = "XYZ".index(up[2])
                tol = 1e-6 * max(1.0, abs(lo), abs(hi))
                picked = {n for n, p in self.nodes.items() if lo - tol <= p[axis] <= hi + tol}
            else:
                raise ValueError(f"unsupported NSEL item {up[1]!r}")
            self.nsel = self._select(self.nsel, self.nodes, up[0], picked)
        elif name == "ESEL":
            if up[0] in ("ALL", "NONE"):
                self.esel = None if up[0] == "ALL" else set()
                return
            lo = int(self.num(a[3]))
            hi = int(self.num(a[4], lo)) if a[4].strip() else lo
            inc = int(self.num(a[5], 1)) or 1
            wanted = set(range(lo, hi + 1, inc))
            key = {
                "ELEM": None,
                "": None,
                "SEC": "sec",
                "SECN": "sec",
                "TYPE": "type",
                "MAT": "mat",
            }
            if up[1] not in key:
                raise ValueError(f"unsupported ESEL item {up[1]!r}")
            attr = key[up[1]]
            picked = {
                n for n, e in self.elements.items() if (n if attr is None else e[attr]) in wanted
            }
            self.esel = self._select(self.esel, self.elements, up[0], picked)
        elif name == "ALLSEL":
            self.nsel = self.esel = None
        elif name == "CM":
            if up[1].startswith("ELEM"):
                self.components[up[0]] = self.selected_elements()
        elif name == "D":
            labels = [up[1]] + [s for s in up[6:] if s]
            value = self.num(a[2])
            for node in self.node_range(a, 0, 4, 5):
                for label in labels:
                    dofs = range(6) if label == "ALL" else [_DOFS.index(label)]
                    for dof in dofs:
                        self.fixed_dofs.setdefault(node, {})[dof] = value
        elif name == "F":
            if up[1] not in _LOADS:
                raise ValueError(f"unsupported load label {up[1]!r}")
            for node in self.node_range(a, 0, 4, 5):
                self.loads.append((node, up[1], self.num(a[2])))
        elif name == "DSYM":
            if up[0] != "SYMM" or up[1] not in ("X", "Y", "Z"):
                raise ValueError("only DSYM,SYMM,X|Y|Z is supported")
            self.symmetry += [(node, up[1]) for node in self.selected_nodes()]
        elif name == "CERIG":
            if up[2] not in ("ALL", ""):
                raise ValueError("only CERIG with all degrees of freedom is supported")
            master = int(self.num(a[0]))
            slaves = self.selected_nodes() if up[1] == "ALL" else [int(self.num(a[1]))]
            self.rigid.setdefault(master, []).extend(s for s in slaves if s != master)
        else:
            raise NotImplementedError(f"unsupported command {name}")


def _statements(text: str) -> list[tuple[int, str]]:
    out = []
    for number, line in enumerate(text.splitlines(), 1):
        for part in line.split("!", 1)[0].split("$"):
            if part.strip():
                out.append((number, part.strip()))
    return out


def _run(reader: _Reader, stmts: list[tuple[int, str]]) -> None:
    i = 0
    while i < len(stmts):
        number, text = stmts[i]
        head, *args = (s for s in text.split(","))
        name = head.strip().upper()
        try:
            assign = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)", text)
            if assign and "," not in assign.group(1):
                reader.params[assign.group(1).upper()] = reader.num(assign.group(2))
            elif name == "*SET":
                reader.params[args[0].strip().upper()] = reader.num(args[1])
            elif name == "*AFUN":
                reader.degrees = args[0].strip().upper().startswith("DEG")
            elif name == "*DO":
                depth, j = 1, i + 1
                while j < len(stmts) and depth:
                    word = stmts[j][1].split(",")[0].strip().upper()
                    depth += (word == "*DO") - (word == "*ENDDO")
                    j += 1
                if depth:
                    raise ValueError("*DO without *ENDDO")
                var = args[0].strip().upper()
                start, stop = reader.num(args[1]), reader.num(args[2])
                step = reader.num(args[3], 1.0) if len(args) > 3 else 1.0
                count = math.floor((stop - start) / step + 1e-9) + 1
                for k in range(max(count, 0)):
                    reader.params[var] = start + k * step
                    _run(reader, stmts[i + 1 : j - 1])
                i = j
                continue
            elif name.startswith("/") or name in _IGNORED:
                if name not in _SILENT:
                    reader.notes.append(f"line {number}: {text} (ignored)")
            elif name.startswith("*"):
                raise NotImplementedError(f"unsupported command {name}")
            else:
                reader.command(name, list(args))
        except (ApdlError, RecursionError):
            raise
        except Exception as err:  # re-raised with the line number
            raise ApdlError(f"line {number}: {text!r}: {err}") from err
        i += 1


def _chain(elements: dict[int, dict], ids: list[int]) -> tuple[list[int], list[int]]:
    """Order the optimisable elements into one chain: (element ids, node ids)."""
    links: dict[int, list[tuple[int, int]]] = {}
    for eid in ids:
        a, b = elements[eid]["nodes"]
        links.setdefault(a, []).append((eid, b))
        links.setdefault(b, []).append((eid, a))
    ends = sorted(n for n, lk in links.items() if len(lk) == 1)
    if len(ends) != 2 or any(len(lk) > 2 for lk in links.values()):
        raise ApdlError(f"the elements of {OPT_COMPONENT} must form one open chain")
    order, nodes, used = [], [ends[0]], set()
    while len(order) < len(ids):
        step = [(e, n) for e, n in links[nodes[-1]] if e not in used]
        if not step:
            raise ApdlError(f"the elements of {OPT_COMPONENT} are not connected")
        used.add(step[0][0])
        order.append(step[0][0])
        nodes.append(step[0][1])
    return order, nodes


def _beam_section(sec: dict):
    """(factory, parameter names, parameter values) of a script beam section."""
    subtype, data = sec["subtype"], sec["data"]
    if subtype in MAPDL_SUBTYPES:
        cls = MAPDL_SUBTYPES[subtype]
        section = cls.from_secdata(*data)
        return (
            cls.from_params,
            cls.param_names,
            {n: float(getattr(section, n)) for n in cls.param_names},
        )
    if subtype == "ASEC":
        area, iyy, iyz, izz, _, tors = data[:6]
        tkz, tky = (list(data) + [0.0] * 12)[10:12]
        if tkz <= 0.0 or tky <= 0.0:
            raise ApdlError("ASEC needs the overall dimensions TKz, TKy to recover stresses")
        shape = SectionShape.from_properties(
            area, iyy, izz, tors, y_max=tky / 2.0, z_max=tkz / 2.0, iyz=iyz, name="ASEC"
        )
        return shape, ("scale",), {"scale": 1.0}
    raise ApdlError(f"unsupported beam section subtype {subtype!r}")


def read_apdl(
    source: str,
    sigma_y: float,
    component: str = OPT_COMPONENT,
    rho: float | None = None,
    parameters: dict[str, float] | None = None,
) -> ApdlImport:
    """
    Build a ``StructuralModel`` from an APDL script.

    Parameters
    ----------
    source : str
        Path of the script, or the script text itself.
    sigma_y : float
        Yield stress of the material [MPa] (APDL has no such property).
    component : str
        Element component that marks the beam to optimise.
    rho : float | None
        Density [ton/mm³]; defaults to ``MP,DENS`` of the script.
    parameters : dict[str, float] | None
        Values that override nothing but are available to the script as
        predefined parameters (e.g. to drive a parametric script).

    Returns
    -------
    ApdlImport

    Raises
    ------
    ApdlError
        For an unsupported command or an inconsistent model; the message
        gives the line of the script.
    """
    text = source
    if "\n" not in source and os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            text = f.read()
    rd = _Reader(parameters)
    _run(rd, _statements(text))

    name = component.upper()
    if name not in rd.components or not rd.components[name]:
        raise ApdlError(
            f"no element component {component}: mark the beam to optimise with "
            f"ESEL,... and CM,{component},ELEM"
        )
    opt_ids = rd.components[name]

    def is_beam(e: dict) -> bool:
        return rd.etypes.get(e["type"], "") in _BEAMS

    for eid, e in rd.elements.items():
        if is_beam(e) and len(e["nodes"]) != 2:
            raise ApdlError(
                f"element {eid}: beam orientation nodes are not supported "
                "(the default BEAM188 orientation is used)"
            )
        if eid in opt_ids and not is_beam(e):
            raise ApdlError(f"element {eid} of {component} is not a beam")

    order, chain = _chain(rd.elements, opt_ids)
    node_name = {n: f"n{n}" for n in rd.nodes}
    mesh = BeamMesh(
        np.array([rd.nodes[n] for n in chain]),
        anchors={node_name[n]: i for i, n in enumerate(chain)},
    )
    points = {node_name[n]: p for n, p in rd.nodes.items() if n not in chain}

    # Optimisable sections: one subtype, parameters per element
    secs = [rd.sections[rd.elements[e]["sec"]] for e in order]
    if len({s["subtype"] for s in secs}) != 1:
        raise ApdlError(f"the elements of {component} must share one section subtype")
    factory, names, _ = _beam_section(secs[0])
    values = [_beam_section(s)[2] for s in secs]
    initial = {n: np.array([v[n] for v in values]) for n in names}

    # Material
    mats = {rd.elements[e]["mat"] for e in rd.elements}
    if len(mats) != 1:
        raise ApdlError("models with more than one material are not supported")
    mp = rd.materials.get(mats.pop(), {})
    density = rho if rho is not None else mp.get("DENS")
    if "EX" not in mp or density is None:
        raise ApdlError("the material needs MP,EX and MP,DENS (or the rho argument)")
    material = Material(mp["EX"], mp.get("PRXY", mp.get("NUXY", 0.3)), density, sigma_y)

    # Fixed elements
    fixed = []
    for eid in sorted(set(rd.elements) - set(opt_ids)):
        e = rd.elements[eid]
        sec = rd.sections[e["sec"]]
        names_e = [node_name[n] for n in e["nodes"]]
        if is_beam(e):
            fac, _, vals = _beam_section(sec)
            fixed.append(FixedBeam(names_e[0], names_e[1], fac(vals)))
        elif rd.etypes.get(e["type"], "") in _SHELLS:
            corners = names_e if len(names_e) == 4 else [*names_e, names_e[-1]]
            fixed.append(FixedShell(corners, thickness=sec["data"][0]))
        else:
            raise ApdlError(f"element {eid}: unsupported element type {rd.etypes.get(e['type'])}")

    # Boundary conditions
    bcs: list = []
    for node, normal in rd.symmetry:
        bcs.append(Symmetry(node_name[node], normal))
    for master, slaves in rd.rigid.items():
        bcs.append(RigidRegion(node_name[master], [node_name[s] for s in slaves]))
    for node, dofs in sorted(rd.fixed_dofs.items()):
        if len(dofs) == 6 and not any(dofs.values()):
            bcs.append(Clamp(node_name[node]))
        else:
            bcs += [
                ImposedDisplacement(node_name[node], _DOFS[d], v) for d, v in sorted(dofs.items())
            ]
    for node, label, value in rd.loads:
        bcs.append(NodalForce(node_name[node], label, value))

    model = StructuralModel(
        OptimizableBeam(mesh, section_factory=factory), material, points, fixed, bcs
    )
    return ApdlImport(model, factory, tuple(names), initial, order, rd.notes)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


_WRITER_NAMES = {"prep7": "/PREP7", "clear": "/CLEAR", "filname": "/FILNAME", "finish": "FINISH"}


class _ApdlWriter:
    """Stands in for a MAPDL session and records the commands as APDL text."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    @staticmethod
    def _fmt(value) -> str:
        if isinstance(value, bool | str):
            return str(value)
        if isinstance(value, int | np.integer):
            return str(int(value))
        return f"{float(value):.12g}"

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        command = _WRITER_NAMES.get(name, name.upper())

        def record(*args):
            self.lines.append(",".join([command, *(self._fmt(a) for a in args)]).rstrip(","))

        return record


def write_apdl(
    model: StructuralModel, law: SectionLaw, path: str | None = None, title: str = ""
) -> str:
    """
    Write a design as an APDL input file.

    The commands are exactly those the framework sends to MAPDL
    (``ModelBuilder``), followed by the ``OPT_BEAM`` component and a linear
    static solution block.

    Parameters
    ----------
    model : StructuralModel
        Structure.
    law : SectionLaw
        Section law of the optimisable beam, e.g.
        ``problem.law_from_x(result.x)``.
    path : str | None
        File to write; ``None`` only returns the text.
    title : str
        ``/TITLE`` of the analysis.

    Returns
    -------
    str
        The script.
    """
    writer = _ApdlWriter()
    ModelBuilder(writer).build(model, law)
    body = [ln for ln in writer.lines if ln not in ("/CLEAR", "FINISH")]
    n_elem = model.beam.mesh.n_elem
    lines = [
        "! Written by beam-opt-framework (units: mm, N, ton, s, MPa)",
        f"! Optimisable beam: elements 1 to {n_elem}, component {OPT_COMPONENT}",
        "/CLEAR",
        f"/TITLE,{title}" if title else "/TITLE,beam-opt-framework design",
        *body,
        f"ESEL,S,ELEM,,1,{n_elem}",
        f"CM,{OPT_COMPONENT},ELEM",
        "ALLSEL,ALL",
        "FINISH",
        "/SOLU",
        "ANTYPE,STATIC",
        "SOLVE",
        "FINISH",
    ]
    text = "\n".join(lines) + "\n"
    if path:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
