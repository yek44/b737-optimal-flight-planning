"""Per-flight result dataclass + npz/json persistence.

Every method (direct, rl) returns a `FlightResult` so the comparison
script can load and overlay them uniformly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class FlightResult:
    """A single flight's solution from one method.

    Conventions:
        - States stored in flat-Earth ENU + SI.
        - Time vector `t` is monotone non-decreasing.
        - `u` may have shape (N-1, 3) for collocation or (N, 3) for
          piecewise-constant; consumers should treat both.
    """
    method: str             # "direct" | "rl"
    flight_id: int          # 1, 2, 3
    t: np.ndarray           # (N,) seconds
    state: np.ndarray       # (N, 6) -- (x, y, h, v, psi, m) in SI/ENU
    control: np.ndarray     # (N or N-1, 3) -- (gamma, mu, delta)
    fuel_total_kg: float
    duration_s: float
    cost: float
    converged: bool
    origin_lambda_deg: float
    origin_phi_deg: float
    solver_meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, outdir: Path | str) -> None:
        """Write `solution.npz` + `meta.json` under `outdir`."""
        d = Path(outdir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "solution.npz",
            t=self.t,
            state=self.state,
            control=self.control,
        )
        meta = {
            "method": self.method,
            "flight_id": self.flight_id,
            "fuel_total_kg": self.fuel_total_kg,
            "duration_s": self.duration_s,
            "cost": self.cost,
            "converged": self.converged,
            "origin_lambda_deg": self.origin_lambda_deg,
            "origin_phi_deg": self.origin_phi_deg,
            "solver_meta": self.solver_meta,
            "n_states": int(self.state.shape[0]),
            "n_controls": int(self.control.shape[0]),
        }
        with open(d / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, indir: Path | str) -> "FlightResult":
        """Load a previously-saved result from a directory."""
        d = Path(indir)
        npz = np.load(d / "solution.npz")
        with open(d / "meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return cls(
            method=meta["method"],
            flight_id=meta["flight_id"],
            t=npz["t"],
            state=npz["state"],
            control=npz["control"],
            fuel_total_kg=meta["fuel_total_kg"],
            duration_s=meta["duration_s"],
            cost=meta["cost"],
            converged=meta["converged"],
            origin_lambda_deg=meta["origin_lambda_deg"],
            origin_phi_deg=meta["origin_phi_deg"],
            solver_meta=meta.get("solver_meta", {}),
        )

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"[{self.method}] flight {self.flight_id} | "
            f"converged={self.converged} | "
            f"cost={self.cost:.3e} | "
            f"fuel={self.fuel_total_kg:.1f} kg | "
            f"duration={self.duration_s / 60.0:.1f} min"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Dataclass -> dict (for JSON; arrays are dropped)."""
        d = asdict(self)
        d.pop("t", None)
        d.pop("state", None)
        d.pop("control", None)
        return d
