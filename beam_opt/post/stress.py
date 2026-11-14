"""
beam_opt.post.stress
====================
Per-element equivalent stress of the optimisable beam.

``SectionStressEvaluator`` — von Mises stress for any cross-section, from the
BEAM188 internal forces and the check points of the ``Section`` (default).

``RectBeamStressEvaluator`` — the 0.1 evaluator for solid rectangular
sections, kept for comparison; see its differences in ``SectionStressEvaluator``.

Legacy rectangular evaluator
----------------------------

For each optimisable element the stress field is sampled at three locations
along the element axis (node I, element mid-point, node J) and at several
points around the rectangular cross-section perimeter.  The cross-section
geometry is evaluated *locally* at each of the three positions using the
current section law, so the formula is consistent with a varying b(s)/h(s).

Stress components extracted from MAPDL via ETABLE (SMISC items, see
Table 188.2 of the ANSYS element reference):
    SMISC 4   – Torsional moment TQ at node I
    SMISC 5   – Shear force SFZ at node I
    SMISC 6   – Shear force SFY at node I
    SMISC 17  – Torsional moment TQ at node J
    SMISC 18  – Shear force SFZ at node J
    SMISC 19  – Shear force SFY at node J
    SMISC 31  – sd  at node I (direct stress)
    SMISC 32  – byt at node I (+y bending top fibre)
    SMISC 33  – byb at node I (−y bending bottom fibre)
    SMISC 34  – bzt at node I (+z bending top fibre)
    SMISC 35  – bzb at node I (−z bending bottom fibre)
    SMISC 36–40 – same quantities at node J

Shear stresses are estimated with the Jourawski formula (max = 1.5 * V / A)
and Saint-Venant torsion (max = Mt / Wt), with A and Wt from the ``Section``.

The Jourawski maximum of SFZ lies on the line z = 0, i.e. at the mid-side of
the ±Y faces, so it is combined with the SByT/SByB fibres; likewise SFY is
combined with SBzT/SBzB.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .extract import beam_smisc

# SMISC identifiers used by the legacy rectangular evaluator
_SMISC_IDS = [4, 5, 6, 17, 18, 19, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# Internal forces of BEAM188 (Table 188.2): FX, MY, MZ, TQ, SFZ, SFY at node I
# (SMISC 1–6) and at node J (SMISC 14–19).
_FIRST_RESULTANT = (1, 14)
_RESULTANT_IDS = [first + k for first in _FIRST_RESULTANT for k in range(6)]


class StressEvaluator(ABC):
    """
    Interface for per-element equivalent-stress evaluators.

    Implementors receive the live MAPDL session (already in ``/POST1`` with
    the last results set loaded), the ``OptimizableBeam`` and the current
    ``SectionLaw``, and return one scalar per optimisable element.
    """

    @abstractmethod
    def __call__(self, mapdl, beam, law) -> np.ndarray:
        """
        Compute the element-wise equivalent stress field.

        Parameters
        ----------
        mapdl
            Active MAPDL session, in /POST1 with the LAST set loaded.
        beam : OptimizableBeam
            Optimisable beam (mesh and section factory).
        law : SectionLaw
            Current section law.

        Returns
        -------
        ndarray, shape (n_elem,)
            Equivalent stress for each optimisable element [MPa].
        """


class SectionStressEvaluator(StressEvaluator):
    """
    Von Mises stress for any ``Section``, from the element internal forces.

    At both nodes of every element, and at every check point of the section
    (``Section.stress_points``):

        σ_x  = N/A + My·z/Iy − Mz·y/Iz
        τ    = vector sum of the torsion and shear-force stresses
        σ_VM = sqrt(σ_x² + 3 τ²)

    and the element value is the maximum.  The section is the one MAPDL
    uses for the element (the section law at the element mid-point), so the
    direct and bending stresses coincide with the BEAM188 output.

    Differences from ``RectBeamStressEvaluator`` on a rectangle:

    * corners: no torsion stress (it vanishes there); the legacy evaluator
      added the maximum torsion stress, overestimating;
    * mid-sides: torsion and shear stresses are parallel and add linearly;
      the legacy evaluator combined them as orthogonal, underestimating;
    * no extra candidate at the element mid-length.

    Without torsion the two coincide wherever a corner or a mid-side governs.
    """

    def __call__(self, mapdl, beam, law) -> np.ndarray:
        return self.from_forces(beam_smisc(mapdl, beam.mesh.n_elem, _RESULTANT_IDS), beam, law)

    def from_forces(self, sm: dict[int, np.ndarray], beam, law) -> np.ndarray:
        """
        Element stress from the internal forces, whatever solver produced them.

        Parameters
        ----------
        sm : dict[int, ndarray]
            Internal forces keyed by BEAM188 SMISC item: 1–6 at node I and
            14–19 at node J (FX, MY, MZ, TQ, SFZ, SFY) [N], [N·mm].
        beam : OptimizableBeam
            Optimisable beam.
        law : SectionLaw
            Current section law.

        Returns
        -------
        ndarray, shape (n_elem,)
            Equivalent stress per element [MPa].
        """
        mesh = beam.mesh
        section = beam.sections_at(law, mesh.s_elem)
        pts = section.stress_points()
        area, iy, iz = section.area, section.iy, section.iz

        sigma = np.zeros(mesh.n_elem)
        for first in _FIRST_RESULTANT:
            n, my, mz, tq, vz, vy = (sm[first + k] for k in range(6))
            sx = n / area + my * pts.z / iy - mz * pts.y / iz
            txy = pts.ty_t * tq + pts.ty_vy * vy + pts.ty_vz * vz
            txz = pts.tz_t * tq + pts.tz_vy * vy + pts.tz_vz * vz
            vm = np.sqrt(sx**2 + 3.0 * (txy**2 + txz**2))  # (n_points, n_elem)
            sigma = np.maximum(sigma, vm.max(axis=0))
        return sigma


class RectBeamStressEvaluator(StressEvaluator):
    """Legacy (0.1) von Mises evaluator for rectangular BEAM188 elements."""

    def __call__(self, mapdl, beam, law) -> np.ndarray:
        mesh = beam.mesh
        sm = beam_smisc(mapdl, mesh.n_elem, _SMISC_IDS)

        # Internal forces / moments at node I and node J
        Mt_I, SFz_I, SFy_I = sm[4], sm[5], sm[6]
        Mt_J, SFz_J, SFy_J = sm[17], sm[18], sm[19]

        # Direct + bending stresses at node I and node J
        sd_I, byt_I, byb_I, bzt_I, bzb_I = (sm[k] for k in range(31, 36))
        sd_J, byt_J, byb_J, bzt_J, bzb_J = (sm[k] for k in range(36, 41))

        # ---------------------------------------------------------------
        # Section properties at I, mid-point, J
        # ---------------------------------------------------------------
        sec_I = beam.sections_at(law, mesh.s[:-1])
        sec_M = beam.sections_at(law, mesh.s_elem)
        sec_J = beam.sections_at(law, mesh.s[1:])
        A_I, Wt_I = sec_I.area, sec_I.torsion_modulus
        A_M, Wt_M = sec_M.area, sec_M.torsion_modulus
        A_J, Wt_J = sec_J.area, sec_J.torsion_modulus

        # ---------------------------------------------------------------
        # Shear + torsion stresses
        # ---------------------------------------------------------------
        # Mid-point resultants: average of the two end values
        Mt_M = 0.5 * (Mt_I + Mt_J)
        SFy_M = 0.5 * (SFy_I + SFy_J)
        SFz_M = 0.5 * (SFz_I + SFz_J)

        tau_T_I = np.abs(Mt_I) / Wt_I
        tau_T_M = np.abs(Mt_M) / Wt_M
        tau_T_J = np.abs(Mt_J) / Wt_J

        # Jourawski: τ_max = 1.5 V / A
        tau_Vy_I = 1.5 * np.abs(SFy_I) / A_I
        tau_Vz_I = 1.5 * np.abs(SFz_I) / A_I
        tau_Vy_M = 1.5 * np.abs(SFy_M) / A_M
        tau_Vz_M = 1.5 * np.abs(SFz_M) / A_M
        tau_Vy_J = 1.5 * np.abs(SFy_J) / A_J
        tau_Vz_J = 1.5 * np.abs(SFz_J) / A_J

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

        candidates.append(_vm(sd_I + bzt_I, np.hypot(tau_Vy_I, tau_T_I)))
        candidates.append(_vm(sd_I + bzb_I, np.hypot(tau_Vy_I, tau_T_I)))
        candidates.append(_vm(sd_I + byt_I, np.hypot(tau_Vz_I, tau_T_I)))
        candidates.append(_vm(sd_I + byb_I, np.hypot(tau_Vz_I, tau_T_I)))

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

        candidates.append(_vm(sd_J + bzt_J, np.hypot(tau_Vy_J, tau_T_J)))
        candidates.append(_vm(sd_J + bzb_J, np.hypot(tau_Vy_J, tau_T_J)))
        candidates.append(_vm(sd_J + byt_J, np.hypot(tau_Vz_J, tau_T_J)))
        candidates.append(_vm(sd_J + byb_J, np.hypot(tau_Vz_J, tau_T_J)))

        vm_matrix = np.column_stack(candidates)  # shape (n_elem, n_candidates)
        return vm_matrix.max(axis=1)


def _vm(sx: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress: sqrt(σ² + 3τ²)."""
    return np.sqrt(sx**2 + 3.0 * tau**2)
