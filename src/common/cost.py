"""Mayer + Lagrange cost terms for the optimal-control problem.

PDF eq. (7):
    J = (x(tf) - xd)^2 + (y(tf) - yd)^2 + (h(tf) - hd)^2
        + integral_0^tf (0.05 + f) dt

The terminal-distance Mayer term uses ENU meters squared (the
destination is converted to ENU once per flight). The integrand is
fuel flow [kg/s] plus a small constant 0.05 that penalises duration.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .aircraft import AircraftParams, fuel_flow

# Per-second time penalty from the PDF integrand.
TIME_PENALTY: float = 0.05


def lagrange_integrand(
    v_mps: Any,
    h_m: Any,
    delta: Any,
    ac: AircraftParams,
    math: Any = np,
) -> Any:
    """Running cost: 0.05 + f(v, h, delta) [kg/s + 1/s units]."""
    return TIME_PENALTY + fuel_flow(v_mps, h_m, delta, ac, math)


def mayer_terminal(
    x_tf_m: Any,
    y_tf_m: Any,
    h_tf_m: Any,
    xd_m: Any,
    yd_m: Any,
    hd_m: Any,
) -> Any:
    """Terminal Mayer cost: squared ENU + altitude error to destination.

    Args:
        x_tf_m, y_tf_m: Terminal east/north position [m].
        h_tf_m: Terminal altitude [m].
        xd_m, yd_m, hd_m: Destination east/north position [m] and altitude [m].

    Returns:
        Squared distance [m^2 + m^2 + m^2].
    """
    dx = x_tf_m - xd_m
    dy = y_tf_m - yd_m
    dh = h_tf_m - hd_m
    return dx * dx + dy * dy + dh * dh
