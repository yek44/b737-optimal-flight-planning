"""YAML config loaders for flight plans, aircraft, and wind field."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml  # type: ignore[import-untyped]

from .aircraft import AircraftParams
from .wind_field import WindField


@dataclass(frozen=True)
class InitialState:
    """Initial conditions for a flight (PDF Table 3 left columns)."""
    lambda_deg: float
    phi_deg: float
    h_m: float
    v_mps: float
    psi_deg: float
    m_kg: float


@dataclass(frozen=True)
class Destination:
    """Destination waypoint for a flight (PDF Table 3 right columns)."""
    lambda_deg: float
    phi_deg: float
    h_m: float


@dataclass(frozen=True)
class FlightPlan:
    """A single flight's specification."""
    id: int
    initial: InitialState
    destination: Destination


# Repository root (parent of `src/`)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_aircraft(path: Path | str | None = None) -> AircraftParams:
    """Load `data/aircraft_b738.yaml` into an AircraftParams instance."""
    p = Path(path) if path else _REPO_ROOT / "data" / "aircraft_b738.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    return AircraftParams.from_dict(d)


def load_wind_field(path: Path | str | None = None) -> WindField:
    """Load `data/wind_field.yaml` into a WindField instance."""
    p = Path(path) if path else _REPO_ROOT / "data" / "wind_field.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    return WindField.from_dict(d)


def load_flight_plans(path: Path | str | None = None) -> List[FlightPlan]:
    """Load `data/flight_plans.yaml` into a list of FlightPlan."""
    p = Path(path) if path else _REPO_ROOT / "data" / "flight_plans.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    out: List[FlightPlan] = []
    for f in d["flights"]:
        out.append(
            FlightPlan(
                id=int(f["id"]),
                initial=InitialState(**f["initial"]),
                destination=Destination(**f["destination"]),
            )
        )
    return out


def get_flight_plan(flight_id: int, path: Path | str | None = None) -> FlightPlan:
    """Load a specific flight plan by integer id (1, 2, or 3)."""
    plans = load_flight_plans(path)
    for fp in plans:
        if fp.id == flight_id:
            return fp
    raise ValueError(f"No flight plan with id={flight_id}")
