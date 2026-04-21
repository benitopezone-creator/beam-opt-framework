"""
beam_opt.stress.rect_beam
=========================
Von Mises stress evaluator for BEAM188 elements with rectangular cross-sections.

Theory summary
--------------
For each optimisable element the stress field is sampled at three locations
along the element axis (node I, element mid-point, node J) and at several
points around the rectangular cross-section perimeter.  The cross-section
geometry is evaluated *locally* at each of the three positions using the
current ``SectionLaw``, so the formula is consistent with a varying b(s)/h(s).

Stress components extracted from MAPDL via ETABLE (SMISC items):
    SMISC 4   – Torsional moment Mt
    SMISC 5   – Shear force SFy
    SMISC 6   – Shear force SFz
    SMISC 31  – sd  at node I (direct stress)
    SMISC 32  – byt at node I (+y bending top fibre)
    SMISC 33  – byb at node I (−y bending bottom fibre)
    SMISC 34  – bzt at node I (+z bending top fibre)
    SMISC 35  – bzb at node I (−z bending bottom fibre)
    SMISC 36–40 – same quantities at node J

Shear stresses are estimated with the Jourawski formula (max = 1.5 * V / A)
and Saint-Venant torsion (max = Mt / Wt) where Wt is computed from the
classical series expansion for a rectangle.
"""

from __future__ import annotations

import numpy as np

from .base import AbstractStressEvaluator


# SMISC identifiers used by the evaluator
_SMISC_IDS = [4, 5, 6, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]


class RectBeamStressEvaluator(AbstractStressEvaluator):
    """
    Conservative von Mises evaluator for rectangular BEAM188 elements.

    No constructor arguments are required; all parameters are derived at
    call time from the ``model`` and ``law`` objects.
    """

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def __call__(self, mapdl, model, law) -> np.ndarray:
        """
        Compute per-element von Mises stress.

        The method assumes that ``mapdl`` is already in ``/POST1`` with the
        last solution set selected.

        Returns
        -------
        vm_per_elem : ndarray, shape (n_elem,)
        """
        # Select only optimisable elements (section numbers 1 … n_elem)
        mapdl.esel("S", "SECN", "", 1, model.n_elem)

        # ---------------------------------------------------------------
        # Populate ETABLE entries
        # ---------------------------------------------------------------
        for sid in _SMISC_IDS:
            mapdl.etable(f"SM{sid}", "SMISC", sid)

        n = model.n_elem
        elem_ids = np.arange(1, n + 1)

        def _get(sid: int) -> np.ndarray:
            return np.array(
                [mapdl.get_value("ELEM", int(eid), "ETAB", f"SM{sid}") for eid in elem_ids],
                dtype=float,
            )

        # Internal forces / moments
        Mt  = _get(4)
        SFy = _get(5)
        SFz = _get(6)

        # Direct + bending stresses at node I
        sd_I  = _get(31)
        byt_I = _get(32)
        byb_I = _get(33)
        bzt_I = _get(34)
        bzb_I = _get(35)

        # Direct + bending stresses at node J
        sd_J  = _get(36)
        byt_J = _get(37)
        byb_J = _get(38)
        bzt_J = _get(39)
        bzb_J = _get(40)

        # ---------------------------------------------------------------
        # Section geometry at I, mid-point, J
        # ---------------------------------------------------------------
        b_I, h_I = law.eval(model.s[:-1])
        b_M, h_M = law.eval(model.s_elem)
        b_J, h_J = law.eval(model.s[1:])

        A_I, Wt_I = _rect_properties(b_I, h_I)
        A_M, Wt_M = _rect_properties(b_M, h_M)
        A_J, Wt_J = _rect_properties(b_J, h_J)

        # ---------------------------------------------------------------
        # Shear + torsion stresses
        # ---------------------------------------------------------------
        tau_T_I = np.abs(Mt) / Wt_I
        tau_T_M = np.abs(Mt) / Wt_M
        tau_T_J = np.abs(Mt) / Wt_J

        # Jourawski: τ_max = 1.5 V / A
        tau_Vy_I = 1.5 * np.abs(SFy) / A_I
        tau_Vz_I = 1.5 * np.abs(SFz) / A_I
        tau_Vy_M = 1.5 * np.abs(SFy) / A_M
        tau_Vz_M = 1.5 * np.abs(SFz) / A_M
        tau_Vy_J = 1.5 * np.abs(SFy) / A_J
        tau_Vz_J = 1.5 * np.abs(SFz) / A_J

        # ---------------------------------------------------------------
        # Von Mises candidates
        # ---------------------------------------------------------------
        candidates = []

        # --- Node I ---
        for sx in [
            sd_I + byt_I + bzt_I,
            sd_I + byt_I + bzb_I,
            sd_I + byb_I + bzt_I,
            sd_I + byb_I + bzb_I,
        ]:
            candidates.append(_vm(sx, tau_T_I))

        candidates.append(_vm(sd_I + bzt_I, np.hypot(tau_Vz_I, tau_T_I)))
        candidates.append(_vm(sd_I + bzb_I, np.hypot(tau_Vz_I, tau_T_I)))
        candidates.append(_vm(sd_I + byt_I, np.hypot(tau_Vy_I, tau_T_I)))
        candidates.append(_vm(sd_I + byb_I, np.hypot(tau_Vy_I, tau_T_I)))

        # --- Mid-point (no extrapolated bending components available) ---
        tau_centro_M = np.sqrt(tau_Vy_M**2 + tau_Vz_M**2 + tau_T_M**2)
        candidates.append(_vm(0.5 * (sd_I + sd_J), tau_centro_M))

        # --- Node J ---
        for sx in [
            sd_J + byt_J + bzt_J,
            sd_J + byt_J + bzb_J,
            sd_J + byb_J + bzt_J,
            sd_J + byb_J + bzb_J,
        ]:
            candidates.append(_vm(sx, tau_T_J))

        candidates.append(_vm(sd_J + bzt_J, np.hypot(tau_Vz_J, tau_T_J)))
        candidates.append(_vm(sd_J + bzb_J, np.hypot(tau_Vz_J, tau_T_J)))
        candidates.append(_vm(sd_J + byt_J, np.hypot(tau_Vy_J, tau_T_J)))
        candidates.append(_vm(sd_J + byb_J, np.hypot(tau_Vy_J, tau_T_J)))

        vm_matrix = np.column_stack(candidates)   # shape (n_elem, n_candidates)
        return vm_matrix.max(axis=1)


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def _rect_properties(b: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Geometric properties of a rectangular cross-section *b × h*.

    Returns
    -------
    A  : cross-sectional area [mm²]
    Wt : Saint-Venant torsional resistance modulus [mm³]
         such that τ_T,max ≈ |Mt| / Wt
    """
    A = b * h
    b_min = np.minimum(b, h)
    b_max = np.maximum(b, h)
    beta = b_min / b_max

    # Saint-Venant torsional constant (series expansion, Timoshenko & Goodier)
    Jt = (b_min * b_max**3 / 3.0) * (1.0 - 0.63 * beta + 0.052 * beta**5)
    Wt = Jt / (b_min / 2.0)
    return A, Wt


def _vm(sx: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress: sqrt(σ² + 3τ²)."""
    return np.sqrt(sx**2 + 3.0 * tau**2)
