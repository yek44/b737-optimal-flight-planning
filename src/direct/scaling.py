"""Nondimensionalization scales used by the direct-collocation NLP.

IPOPT converges far better when decision variables are O(1). We use
fixed scale factors (per CLAUDE.md) rather than per-flight scaling so
warm-starting between flights is straightforward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scales:
    """Fixed scale factors for nondimensionalization.

    Convention: a *scaled* variable `x_s` relates to the physical
    variable `x` by `x = x_s * scale`. So passing `x / scale` to IPOPT
    makes the decision variable O(1).
    """
    x_m: float = 1.0e5      # east position [m]
    y_m: float = 1.0e5      # north position [m]
    h_m: float = 1.0e4      # altitude [m]
    v_mps: float = 250.0    # airspeed [m/s]
    psi_rad: float = 1.0    # heading [rad] -- already O(1)
    m_kg: float = 1.0e5     # mass [kg]
    t_s: float = 3.6e3      # time [s]

    # Controls
    gamma_rad: float = 1.0  # already O(0.1) rad
    mu_rad: float = 1.0     # already O(0.5) rad
    delta: float = 1.0      # already in [0, 1]

    def state_array(self) -> np.ndarray:
        """6-vector of state scales in dynamics order."""
        return np.array(
            [self.x_m, self.y_m, self.h_m, self.v_mps, self.psi_rad, self.m_kg],
            dtype=float,
        )

    def control_array(self) -> np.ndarray:
        """3-vector of control scales (gamma, mu, delta)."""
        return np.array([self.gamma_rad, self.mu_rad, self.delta], dtype=float)


SCALES = Scales()
