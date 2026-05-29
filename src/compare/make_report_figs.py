"""Build the cross-method overlay figures and summary tables.

Reads `results/{direct,rl}/flight_<id>/` and writes:
  results/compare/flight_<id>/{xy, h_time, ...}.png   (overlays)
  results/compare/summary.csv                          (table)
  results/compare/summary.md                           (rendered table)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np

from ..common.coordinates import LLAOrigin, lla_to_enu
from ..common.io_config import load_aircraft, load_flight_plans
from ..common.plotting import overlay_methods
from ..common.results import FlightResult


METHODS = ("direct", "rl")


def _safe_load(method: str, flight_id: int,
               results_root: Path) -> FlightResult | None:
    path = results_root / method / f"flight_{flight_id}"
    if not (path / "meta.json").exists():
        return None
    return FlightResult.load(path)


def _trajectory_metrics(fr: FlightResult, fp) -> dict:
    """Compute report-table metrics from a saved trajectory."""
    origin = LLAOrigin(fp.initial.lambda_deg, fp.initial.phi_deg)
    xd, yd = lla_to_enu(fp.destination.lambda_deg,
                        fp.destination.phi_deg, origin)
    dest = np.array([float(xd), float(yd), fp.destination.h_m])
    state = np.asarray(fr.state)
    dxy = np.linalg.norm(state[:, :2] - dest[:2], axis=1)
    h_err = state[:, 2] - dest[2]
    final_idx = state.shape[0] - 1
    closest_idx = int(np.argmin(dxy))

    terminal_radius = float(fr.solver_meta.get("terminal_radius_m", 500.0))
    terminal_h_tol = float(fr.solver_meta.get("terminal_h_tol_m", 300.0))
    goal_violation = np.hypot(
        np.maximum(dxy - terminal_radius, 0.0),
        np.maximum(np.abs(h_err) - terminal_h_tol, 0.0),
    )
    return {
        "closest_xy_km": float(dxy[closest_idx] / 1000.0),
        "closest_h_err_m": float(h_err[closest_idx]),
        "final_xy_km": float(dxy[final_idx] / 1000.0),
        "final_h_err_m": float(h_err[final_idx]),
        "goal_miss_m": float(np.min(goal_violation)),
    }


def main() -> None:
    results_root = Path("results")
    out_root = results_root / "compare"
    out_root.mkdir(parents=True, exist_ok=True)
    ac = load_aircraft()
    flights = load_flight_plans()

    rows: List[dict] = []
    for fp in flights:
        results: List[FlightResult] = []
        for m in METHODS:
            fr = _safe_load(m, fp.id, results_root)
            if fr is not None:
                results.append(fr)
                tm = _trajectory_metrics(fr, fp)
                rows.append({
                    "flight_id": fp.id,
                    "method": m,
                    "fuel_kg": round(fr.fuel_total_kg, 1),
                    "duration_min": round(fr.duration_s / 60.0, 1),
                    "cost": f"{fr.cost:.3e}",
                    "converged": fr.converged,
                    "closest_xy_km": round(tm["closest_xy_km"], 2),
                    "closest_h_err_m": round(tm["closest_h_err_m"], 1),
                    "final_xy_km": round(tm["final_xy_km"], 2),
                    "final_h_err_m": round(tm["final_h_err_m"], 1),
                    "goal_miss_m": round(tm["goal_miss_m"], 1),
                })
        if results:
            outdir = out_root / f"flight_{fp.id}"
            outdir.mkdir(parents=True, exist_ok=True)
            overlay_methods(results, ac, outdir)
            print(f"  flight {fp.id}: overlays -> {outdir} "
                  f"({len(results)} methods)")

    # CSV.
    csv_path = out_root / "summary.csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"  table -> {csv_path}")

    # Markdown table for the report.
    md_path = out_root / "summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| flight | method | fuel [kg] | duration [min] | "
                 "cost J | success | closest xy [km] | h err @ closest [m] | "
                 "goal miss [m] |\n")
        fh.write("|---:|:---|---:|---:|---:|:---:|---:|---:|---:|\n")
        for r in rows:
            fh.write(f"| {r['flight_id']} | {r['method']} | "
                     f"{r['fuel_kg']} | {r['duration_min']} | "
                     f"{r['cost']} | {'yes' if r['converged'] else 'no'} | "
                     f"{r['closest_xy_km']} | {r['closest_h_err_m']} | "
                     f"{r['goal_miss_m']} |\n")
    print(f"  markdown -> {md_path}")


if __name__ == "__main__":
    main()
