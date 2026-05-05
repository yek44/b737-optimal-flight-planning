"""Polynomial wind field from Term Project.pdf Table 2.

The polynomial form (lambda, phi in degrees, output in m/s) is:

    W(lam, phi) = c0 + c1*lam + c2*phi + c3*lam*phi
                + c4*lam^2 + c5*phi^2
                + c6*lam^2*phi + c7*lam*phi^2 + c8*lam^2*phi^2

Two backends: numpy (used by DP, RL, and verification) and the same
expression tree for CasADi (used inside the NLP). The math facade
parameter dispatches them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WindField:
    """9 polynomial coefficients for each wind component.

    Coefficients ordered as [c0, c1, ..., c8].
    """
    Wx_coeffs: Tuple[float, ...]
    Wy_coeffs: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.Wx_coeffs) != 9 or len(self.Wy_coeffs) != 9:
            raise ValueError("Wind coefficients must be length 9")

    @classmethod
    def from_dict(cls, d: dict) -> "WindField":
        """Build from `data/wind_field.yaml` schema."""
        return cls(
            Wx_coeffs=tuple(d["Wx"]),
            Wy_coeffs=tuple(d["Wy"]),
        )


def _polynomial(lam_deg: Any, phi_deg: Any, c: Sequence[float]) -> Any:
    """Evaluate the 9-term polynomial in (lambda_deg, phi_deg)."""
    L = lam_deg
    P = phi_deg
    L2 = L * L
    P2 = P * P
    return (
        c[0]
        + c[1] * L
        + c[2] * P
        + c[3] * L * P
        + c[4] * L2
        + c[5] * P2
        + c[6] * L2 * P
        + c[7] * L * P2
        + c[8] * L2 * P2
    )


def wind(
    lam_deg: Any,
    phi_deg: Any,
    wf: WindField,
    math: Any = np,  # noqa: ARG001 - kept for API symmetry
) -> Tuple[Any, Any]:
    """Wind components at (lambda, phi) [deg].

    Args:
        lam_deg: Longitude [deg]. Scalar / array / CasADi symbol.
        phi_deg: Latitude [deg].
        wf: Wind-field coefficients.
        math: Math backend (unused but kept for API symmetry with the
            other modules; the polynomial uses only +, *, which work on
            both numpy and casadi types natively).

    Returns:
        (Wx, Wy) in [m/s], same type/shape as the inputs.
    """
    Wx = _polynomial(lam_deg, phi_deg, wf.Wx_coeffs)
    Wy = _polynomial(lam_deg, phi_deg, wf.Wy_coeffs)
    return Wx, Wy
