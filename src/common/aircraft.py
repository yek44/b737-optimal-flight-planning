"""Boeing 737-800 BADA-style performance model.

Parameters loaded from `data/aircraft_b738.yaml` (PDF Table 1).
Every public function takes a `math` backend so the same code runs
under numpy and CasADi (for the NLP).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .atmosphere import rho as atm_rho
from .constants import G0, M_TO_FT, MPS_TO_KT


@dataclass(frozen=True)
class AircraftParams:
    """Aircraft performance coefficients (BADA-like).

    The naming mirrors the PDF Table 1 columns. All values dimensional
    per the YAML comments.
    """
    name: str
    Cf_cr: float
    Cf1: float
    Cf2: float
    CT_cr: float
    CT_c1: float
    CT_c2: float
    CT_c3: float
    Cd0: float
    k: float
    S: float
    m_OEW_kg: float
    v_min_mps: float
    v_max_mps: float
    h_min_m: float
    h_max_m: float
    gamma_min_deg: float
    gamma_max_deg: float
    mu_min_deg: float
    mu_max_deg: float
    delta_min: float
    delta_max: float

    @classmethod
    def from_dict(cls, d: dict) -> "AircraftParams":
        """Build from the `data/aircraft_b738.yaml` schema."""
        return cls(
            name=d["name"],
            Cf_cr=d["fuel"]["Cf_cr"],
            Cf1=d["fuel"]["Cf1"],
            Cf2=d["fuel"]["Cf2"],
            CT_cr=d["thrust"]["CT_cr"],
            CT_c1=d["thrust"]["CT_c1"],
            CT_c2=d["thrust"]["CT_c2"],
            CT_c3=d["thrust"]["CT_c3"],
            Cd0=d["aerodynamics"]["Cd0"],
            k=d["aerodynamics"]["k"],
            S=d["aerodynamics"]["S"],
            m_OEW_kg=d["limits"]["m_OEW_kg"],
            v_min_mps=d["limits"]["v_min_mps"],
            v_max_mps=d["limits"]["v_max_mps"],
            h_min_m=d["limits"]["h_min_m"],
            h_max_m=d["limits"]["h_max_m"],
            gamma_min_deg=d["limits"]["gamma_min_deg"],
            gamma_max_deg=d["limits"]["gamma_max_deg"],
            mu_min_deg=d["limits"]["mu_min_deg"],
            mu_max_deg=d["limits"]["mu_max_deg"],
            delta_min=d["limits"]["delta_min"],
            delta_max=d["limits"]["delta_max"],
        )


def thrust_max(h_m: Any, ac: AircraftParams, math: Any = np) -> Any:
    """Maximum (climb) thrust at altitude h.

    BADA cruise-thrust polynomial: the constants CT_c1/CT_c2/CT_c3
    are calibrated in feet, hence the conversion factor 3.28084 from
    meters to feet.

    Args:
        h_m: Altitude [m]. Scalar / array / CasADi symbol.
        ac: Aircraft parameters.
        math: numpy or casadi.

    Returns:
        Maximum available thrust [N].
    """
    h_ft = M_TO_FT * h_m
    return ac.CT_cr * ac.CT_c1 * (
        1.0 - h_ft / ac.CT_c2 + ac.CT_c3 * h_ft * h_ft
    )


def specific_fuel_consumption(v_mps: Any, ac: AircraftParams, math: Any = np) -> Any:
    """Specific fuel-consumption coefficient eta.

    BADA form: eta = (Cf1 / 60000) * (1 + V_kt / Cf2), where the /60000
    folds the kg/(min*kN) -> kg/(s*N) conversion. So `eta * Thr` is in
    kg/s when `Thr` is in N.

    Args:
        v_mps: True airspeed [m/s].
        ac: Aircraft parameters.
        math: numpy or casadi.

    Returns:
        Specific fuel-consumption coefficient [kg/(s*N)].
    """
    v_kt = MPS_TO_KT * v_mps
    return (ac.Cf1 / 60_000.0) * (1.0 + v_kt / ac.Cf2)


def fuel_flow(
    v_mps: Any,
    h_m: Any,
    delta: Any,
    ac: AircraftParams,
    math: Any = np,
) -> Any:
    """Cruise fuel flow rate.

    f = delta * Thr_max(h) * eta(v) * Cf_cr   [kg/s]

    Args:
        v_mps: True airspeed [m/s].
        h_m: Altitude [m].
        delta: Throttle setting in [delta_min, delta_max].
        ac: Aircraft parameters.
        math: numpy or casadi.

    Returns:
        Fuel flow [kg/s] (non-negative for valid delta).
    """
    Tmax = thrust_max(h_m, ac, math)
    eta = specific_fuel_consumption(v_mps, ac, math)
    return delta * Tmax * eta * ac.Cf_cr


def lift_coefficient(
    m_kg: Any,
    v_mps: Any,
    h_m: Any,
    mu_rad: Any,
    ac: AircraftParams,
    math: Any = np,
) -> Any:
    """Lift coefficient assuming the aircraft is in a coordinated banked turn
    where the vertical component of lift balances weight: L*cos(mu) = m*g.

        CL = 2*m*g / (rho * S * v^2 * cos(mu))

    Args:
        m_kg: Mass [kg].
        v_mps: True airspeed [m/s].
        h_m: Altitude [m].
        mu_rad: Bank angle [rad].
        ac: Aircraft parameters.
        math: numpy or casadi.

    Returns:
        Required lift coefficient (dimensionless).
    """
    rho = atm_rho(h_m, math)
    return 2.0 * m_kg * G0 / (rho * ac.S * v_mps * v_mps * math.cos(mu_rad))


def drag_coefficient(CL: Any, ac: AircraftParams) -> Any:
    """Polar drag: CD = Cd0 + k*CL^2."""
    return ac.Cd0 + ac.k * CL * CL


def drag_force(
    m_kg: Any,
    v_mps: Any,
    h_m: Any,
    mu_rad: Any,
    ac: AircraftParams,
    math: Any = np,
) -> Any:
    """Aerodynamic drag force [N]."""
    rho = atm_rho(h_m, math)
    CL = lift_coefficient(m_kg, v_mps, h_m, mu_rad, ac, math)
    CD = drag_coefficient(CL, ac)
    return 0.5 * rho * v_mps * v_mps * ac.S * CD
