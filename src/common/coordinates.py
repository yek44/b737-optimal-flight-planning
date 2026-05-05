"""ENU <-> LLA conversions and great-circle helpers.

We integrate dynamics in flat-Earth East-North-Up meters with a
per-flight reference origin at (lambda0, phi0). The wind polynomial is
parameterized in (lambda_deg, phi_deg) so we convert ENU back to LLA
when querying it (and when plotting the x-y trajectory). Round-trip
error stays sub-meter for the few-hundred-km segments in this project.

All angles in *degrees* at the boundary; internally we use radians.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .constants import DEG_TO_RAD, RAD_TO_DEG, R_EARTH


@dataclass(frozen=True)
class LLAOrigin:
    """Reference point used to define the local ENU frame.

    Attributes:
        lambda0_deg: Reference longitude [deg, east-positive].
        phi0_deg: Reference latitude [deg, north-positive].
    """
    lambda0_deg: float
    phi0_deg: float

    @property
    def cos_phi0(self) -> float:
        """Pre-computed cos(phi0) for the flat-Earth scaling."""
        return float(np.cos(self.phi0_deg * DEG_TO_RAD))


def lla_to_enu(
    lambda_deg: Any,
    phi_deg: Any,
    origin: LLAOrigin,
    math: Any = np,
) -> Tuple[Any, Any]:
    """Convert (lambda, phi) [deg] to local ENU (x_east, y_north) [m].

    Flat-Earth approximation:
        x = R * cos(phi0) * (lambda - lambda0) [rad]
        y = R * (phi - phi0) [rad]
    """
    dlam_rad = (lambda_deg - origin.lambda0_deg) * DEG_TO_RAD
    dphi_rad = (phi_deg - origin.phi0_deg) * DEG_TO_RAD
    x_m = R_EARTH * origin.cos_phi0 * dlam_rad
    y_m = R_EARTH * dphi_rad
    return x_m, y_m


def enu_to_lla(
    x_m: Any,
    y_m: Any,
    origin: LLAOrigin,
    math: Any = np,
) -> Tuple[Any, Any]:
    """Convert local ENU (x_east, y_north) [m] back to (lambda, phi) [deg]."""
    dlam_deg = (x_m / (R_EARTH * origin.cos_phi0)) * RAD_TO_DEG
    dphi_deg = (y_m / R_EARTH) * RAD_TO_DEG
    return origin.lambda0_deg + dlam_deg, origin.phi0_deg + dphi_deg


def haversine_distance(
    lambda1_deg: float,
    phi1_deg: float,
    lambda2_deg: float,
    phi2_deg: float,
) -> float:
    """Great-circle distance between two points [m] (numpy only).

    Used for initial-guess construction (great-circle baseline) and
    sanity checks. Not used inside the NLP.
    """
    lam1, phi1 = lambda1_deg * DEG_TO_RAD, phi1_deg * DEG_TO_RAD
    lam2, phi2 = lambda2_deg * DEG_TO_RAD, phi2_deg * DEG_TO_RAD
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return float(R_EARTH * c)


def initial_bearing(
    lambda1_deg: float,
    phi1_deg: float,
    lambda2_deg: float,
    phi2_deg: float,
) -> float:
    """Initial bearing from point 1 to point 2 [rad, measured from north,
    clockwise positive]. Used to seed psi in the NLP initial guess."""
    lam1, phi1 = lambda1_deg * DEG_TO_RAD, phi1_deg * DEG_TO_RAD
    lam2, phi2 = lambda2_deg * DEG_TO_RAD, phi2_deg * DEG_TO_RAD
    dlam = lam2 - lam1
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return float(np.arctan2(x, y))
