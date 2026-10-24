"""
beam_opt.materials
==================
Linear-elastic isotropic material description.

No material is predefined in the package: applications declare their own
``Material`` in their configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """
    Linear-elastic isotropic material.

    Parameters
    ----------
    E : float
        Young's modulus [MPa].
    nu : float
        Poisson's ratio [-].
    rho : float
        Density [ton/mm³].
    sigma_y : float
        Yield (or proof) stress used by stress constraints [MPa].
    name : str
        Optional label, for reports only.
    """

    E: float
    nu: float
    rho: float
    sigma_y: float
    name: str = ""

    def __post_init__(self) -> None:
        if self.E <= 0 or self.rho <= 0 or self.sigma_y <= 0:
            raise ValueError("E, rho and sigma_y must be positive")
        if not -1.0 < self.nu < 0.5:
            raise ValueError("nu must lie in (-1, 0.5)")
