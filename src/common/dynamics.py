"""6-state aircraft dynamics from Term Project.pdf eqs. (1)-(6).

State vector x = (x_e, y_n, h, v, psi, m) in flat-Earth ENU + SI:
    x_e: east  [m]    relative to per-flight reference origin
    y_n: north [m]    relative to per-flight reference origin
    h:   altitude [m]
    v:   true airspeed [m/s]
    psi: heading [rad] -- IMPORTANT: measured COUNTER-CLOCKWISE FROM
         EAST (math convention), NOT clockwise from north. The PDF
         equations use dx = v*cos(psi), dy = v*sin(psi), so psi=0
         points east, psi=pi/2 points north. Table 3's "psi_0 = 0 deg"
         means initial heading due east.
    m:   mass [kg]

Control vector u = (gamma, mu, delta):
    gamma: flight-path angle [rad]
    mu:    bank angle [rad]
    delta: throttle [-]

Wind enters via Wx, Wy (m/s east/north) which are looked up at the
current (lambda, phi) by converting ENU back to LLA via the per-flight
origin.

The same function works under numpy (forward simulation, RL rollouts)
and CasADi (NLP) by passing the appropriate `math` backend.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .aircraft import AircraftParams, drag_force, fuel_flow, lift_coefficient, thrust_max
from .atmosphere import rho as atm_rho
from .constants import G0, RAD_TO_DEG
from .coordinates import LLAOrigin
from .wind_field import WindField, wind as wind_fn


def dynamics_rhs(
    state: Sequence[Any],
    control: Sequence[Any],
    ac: AircraftParams,
    wf: WindField,
    origin: LLAOrigin,
    math: Any = np,
) -> list[Any]:
    """Right-hand side of the 6-state ODE.

    Args:
        state: Sequence of length 6: (x_e, y_n, h, v, psi, m).
        control: Sequence of length 3: (gamma, mu, delta).
        ac: Aircraft parameters (B737-800).
        wf: Polynomial wind field.
        origin: Per-flight ENU reference origin.
        math: numpy or casadi.

    Returns:
        List of 6 derivatives [dx, dy, dh, dv, dpsi, dm], each in SI.
    """
    x_e, y_n, h, v, psi, m = (state[0], state[1], state[2], state[3], state[4], state[5])
    gamma, mu, delta = control[0], control[1], control[2]

    # ENU -> LLA (degrees) for wind lookup. Inlined here so CasADi sees
    # the symbolic dependency on (x_e, y_n) directly.
    cos_phi0 = origin.cos_phi0
    lambda_deg = origin.lambda0_deg + (x_e / (6_371_000.0 * cos_phi0)) * RAD_TO_DEG
    phi_deg = origin.phi0_deg + (y_n / 6_371_000.0) * RAD_TO_DEG
    Wx, Wy = wind_fn(lambda_deg, phi_deg, wf, math)

    # Aerodynamics
    rho = atm_rho(h, math)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)
    sin_mu = math.sin(mu)
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)

    # Lift / drag (uses the L*cos(mu) = m*g balance => CL closed form)
    CL = lift_coefficient(m, v, h, mu, ac, math)
    CD = ac.Cd0 + ac.k * CL * CL
    drag = 0.5 * rho * v * v * ac.S * CD

    # Engine
    Tmax = thrust_max(h, ac, math)
    T = Tmax * delta

    # Fuel flow (kg/s)
    f = fuel_flow(v, h, delta, ac, math)

    # State derivatives -- matches PDF eqs. (1)-(6).
    dx = v * cos_psi * cos_gamma + Wx
    dy = v * sin_psi * cos_gamma + Wy
    dh = v * sin_gamma
    dv = (T - drag) / m - G0 * sin_gamma
    dpsi = (CL * ac.S * rho * v / (2.0 * m)) * sin_mu / cos_gamma
    dm = -f

    return [dx, dy, dh, dv, dpsi, dm]


def dynamics_np(
    state: np.ndarray,
    control: np.ndarray,
    ac: AircraftParams,
    wf: WindField,
    origin: LLAOrigin,
) -> np.ndarray:
    """numpy-only wrapper that returns a (6,) array (for RL rollouts / RK4)."""
    deriv = dynamics_rhs(list(state), list(control), ac, wf, origin, math=np)
    return np.asarray(deriv, dtype=float)


def fuel_flow_np(
    state: np.ndarray, control: np.ndarray, ac: AircraftParams
) -> float:
    """Convenience: instantaneous fuel flow [kg/s] from a (state, control)."""
    _, _, h, v, _, _ = state
    _, _, delta = control
    return float(fuel_flow(v, h, delta, ac, math=np))


def drag_np(state: np.ndarray, control: np.ndarray, ac: AircraftParams) -> float:
    """Convenience: instantaneous drag [N] from (state, control)."""
    _, _, h, v, _, m = state
    _, mu, _ = control
    return float(drag_force(m, v, h, mu, ac, math=np))


def thrust_np(state: np.ndarray, control: np.ndarray, ac: AircraftParams) -> float:
    """Convenience: instantaneous thrust [N] from (state, control)."""
    _, _, h, _, _, _ = state
    _, _, delta = control
    return float(thrust_max(h, ac, math=np) * delta)
