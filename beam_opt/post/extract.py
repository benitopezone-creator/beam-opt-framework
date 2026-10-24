"""
beam_opt.post.extract
=====================
Vectorised result extraction (``*VGET`` through ``mapdl.get_array``).

No per-entity ``get_value`` loops: one MAPDL round trip per quantity.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import numpy as np


@contextmanager
def last_results(mapdl) -> Iterator:
    """
    Enter ``/POST1`` on the last results set; restore selection and leave on exit.

    Examples
    --------
    >>> with last_results(mapdl):
    ...     sm = beam_smisc(mapdl, n_elem, [4, 5, 6])
    """
    mapdl.post1()
    mapdl.set("LAST")
    try:
        yield mapdl
    finally:
        mapdl.allsel("ALL")
        mapdl.finish()


def beam_smisc(mapdl, n_elem: int, smisc_ids: Iterable[int]) -> dict[int, np.ndarray]:
    """
    BEAM188 ``SMISC`` items of the optimisable elements ``1 .. n_elem``.

    Must be called in ``/POST1`` with a results set loaded.  Leaves the
    optimisable elements selected.

    Parameters
    ----------
    mapdl
        MAPDL session (or a test double).
    n_elem : int
        Number of optimisable elements (element and section numbers 1..n_elem,
        as assigned by ``ModelBuilder``).
    smisc_ids : iterable of int
        ``SMISC`` item numbers (Table 188.2 of the ANSYS element reference).

    Returns
    -------
    dict[int, ndarray]
        ``{smisc_id: values}``, each of shape ``(n_elem,)`` (units of the item:
        [N], [N·mm] or [MPa]).
    """
    mapdl.esel("S", "SECN", "", 1, n_elem)
    out: dict[int, np.ndarray] = {}
    for sid in smisc_ids:
        label = f"SM{sid}"
        mapdl.etable(label, "SMISC", sid)
        values = np.asarray(mapdl.get_array("ELEM", 1, "ETAB", label), dtype=float)
        if values.size < n_elem:
            raise RuntimeError(
                f"*VGET of SMISC {sid} returned {values.size} values, expected {n_elem}"
            )
        out[sid] = values[:n_elem]
    return out


def reaction_resultant(mapdl, node_ids: Iterable[int]) -> float:
    """
    Modulus of the resultant reaction force summed over *node_ids* (``FSUM``).

    Must be called in ``/POST1`` with a results set loaded.

    Parameters
    ----------
    mapdl
        MAPDL session (or a test double).
    node_ids : iterable of int
        MAPDL node numbers.

    Returns
    -------
    float
        Resultant reaction modulus [N].
    """
    first, *rest = list(node_ids)
    mapdl.nsel("S", "NODE", "", first)
    for nid in rest:
        mapdl.nsel("A", "NODE", "", nid)
    mapdl.fsum()
    fx = mapdl.get_value("FSUM", 0, "ITEM", "FX")
    fy = mapdl.get_value("FSUM", 0, "ITEM", "FY")
    fz = mapdl.get_value("FSUM", 0, "ITEM", "FZ")
    mapdl.allsel("ALL")
    return float(np.sqrt(fx**2 + fy**2 + fz**2))
