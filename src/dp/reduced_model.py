"""Reduced-state cruise model used by the DP solver.

Full 6-D state grid is computationally hopeless (curse of
dimensionality). We reduce to a 2-D position state `(x_e, y_n)` with
constant cruise altitude/airspeed; mass is treated parametrically
(constant fuel-flow at the cruise condition). The control is the
heading `psi` selected from a discrete set.

Wind is the same polynomial field. Forward step:
    dx = (v*cos(psi) + Wx) * dt
    dy = (v*sin(psi) + Wy) * dt
Per-step cost (0.05 + f) * dt.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..common.aircraft import AircraftParams, fuel_flow


@dataclass(frozen=True)
class CruiseModel:
    """Reduced model parameters frozen at cruise."""
    h_cruise_m: float = 10_000.0
    v_cruise_mps: float = 235.0
    delta_cruise: float = 0.55
    n_headings: int = 32

    def fuel_rate(self, ac: AircraftParams) -> float:
        """Constant fuel-flow [kg/s] at the cruise operating point."""
        return float(fuel_flow(self.v_cruise_mps, self.h_cruise_m,
                               self.delta_cruise, ac))
