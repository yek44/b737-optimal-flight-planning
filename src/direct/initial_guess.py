"""Great-circle baseline trajectory used as the IPOPT initial guess.

The NLP is highly nonlinear; a sensible warm start matters more than
algorithm tweaks. We seed with a straight great-circle ENU line from
start to destination, linear altitude / mass interpolation, constant
cruise airspeed, level controls, and `tf = distance / v_cruise`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.aircraft import AircraftParams, fuel_flow
from ..common.constants import DEG_TO_RAD
from ..common.coordinates import LLAOrigin, lla_to_enu
from ..common.io_config import FlightPlan


@dataclass(frozen=True)
class InitialGuess:
    """Starting point for the NLP.

    Attributes:
        t: (N,) time grid [s] from 0 to `tf`.
        state: (N, 6) state guess in dynamics order.
        control: (N, 3) control guess in (gamma, mu, delta).
        tf: free-final-time guess [s].
        origin: per-flight LLA origin (the *start* point).
        dest_enu: (xd, yd, hd) destination in ENU [m, m, m].
    """
    t: np.ndarray
    state: np.ndarray
    control: np.ndarray
    tf: float
    origin: LLAOrigin
    dest_enu: tuple[float, float, float]


def build_initial_guess(
    fp: FlightPlan,
    ac: AircraftParams,
    n_nodes: int = 61,
    v_cruise_mps: float = 235.0,
) -> InitialGuess:
    """Construct a great-circle baseline guess for a flight plan.

    Args:
        fp: Flight plan (initial conditions + destination).
        ac: Aircraft parameters (used for the fuel-burn estimate).
        n_nodes: Number of collocation nodes (segments = n_nodes - 1).
        v_cruise_mps: Cruise airspeed used for the time / fuel estimate.

    Returns:
        InitialGuess with `state` (N, 6), `control` (N, 3), and `tf`.
    """
    if n_nodes < 3:
        raise ValueError("Need at least 3 nodes for Hermite-Simpson")

    # ENU origin = start point. Destination expressed in this frame.
    origin = LLAOrigin(lambda0_deg=fp.initial.lambda_deg, phi0_deg=fp.initial.phi_deg)
    xd, yd = lla_to_enu(fp.destination.lambda_deg, fp.destination.phi_deg, origin)
    hd = fp.destination.h_m

    # Straight-line ENU distance + heading (ψ measured CCW from east).
    dx = float(xd) - 0.0
    dy = float(yd) - 0.0
    dist_xy = float(np.hypot(dx, dy))
    psi_guess = float(np.arctan2(dy, dx))

    # Time-of-flight guess (ground-track distance / cruise speed).
    tf_guess = max(600.0, dist_xy / v_cruise_mps)

    # Per-second fuel burn at the cruise guess; estimate landing mass.
    f0 = float(fuel_flow(v_cruise_mps, fp.initial.h_m, 0.5, ac))
    m_end = max(ac.m_OEW_kg + 100.0, fp.initial.m_kg - f0 * tf_guess)

    # Linearly-interpolated state guess.
    s = np.linspace(0.0, 1.0, n_nodes)
    x = s * dx
    y = s * dy
    h = (1.0 - s) * fp.initial.h_m + s * hd
    v = np.full(n_nodes, fp.initial.v_mps + s * (v_cruise_mps - fp.initial.v_mps))
    psi = np.full(n_nodes, psi_guess)
    m = (1.0 - s) * fp.initial.m_kg + s * m_end
    state = np.stack([x, y, h, v, psi, m], axis=1)

    # Level controls at half-throttle.
    control = np.zeros((n_nodes, 3))
    control[:, 2] = 0.5  # delta

    # If destination is above start, bias gamma slightly upward.
    if hd > fp.initial.h_m + 100.0:
        control[:, 0] = np.deg2rad(2.0)
    elif hd < fp.initial.h_m - 100.0:
        control[:, 0] = np.deg2rad(-2.0)

    # Override the very first state with the actual initial conditions.
    state[0, 0] = 0.0
    state[0, 1] = 0.0
    state[0, 2] = fp.initial.h_m
    state[0, 3] = fp.initial.v_mps
    state[0, 4] = fp.initial.psi_deg * DEG_TO_RAD
    state[0, 5] = fp.initial.m_kg

    t = np.linspace(0.0, tf_guess, n_nodes)
    return InitialGuess(
        t=t,
        state=state,
        control=control,
        tf=tf_guess,
        origin=origin,
        dest_enu=(float(xd), float(yd), float(hd)),
    )
