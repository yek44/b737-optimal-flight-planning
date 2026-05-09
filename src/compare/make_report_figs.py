"""Build the cross-method overlay figures and summary tables.

Reads `results/{direct,dp,rl}/flight_<id>/` and writes:
  results/compare/flight_<id>/{xy, h_time, ...}.png   (overlays)
  results/compare/summary.csv                          (table)
  results/compare/summary.md                           (rendered table)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from ..common.io_config import load_aircraft, load_flight_plans
from ..common.plotting import overlay_methods
from ..common.results import FlightResult


METHODS = ("direct", "dp", "rl")


def _safe_load(method: str, flight_id: int,
               results_root: Path) -> FlightResult | None:
    path = results_root / method / f"flight_{flight_id}"
    if not (path / "meta.json").exists():
        return None
    return FlightResult.load(path)


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
                rows.append({
                    "flight_id": fp.id,
                    "method": m,
                    "fuel_kg": round(fr.fuel_total_kg, 1),
                    "duration_min": round(fr.duration_s / 60.0, 1),
                    "cost": f"{fr.cost:.3e}",
                    "converged": fr.converged,
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
                 "cost J | converged |\n")
        fh.write("|---:|:---|---:|---:|---:|:---:|\n")
        for r in rows:
            fh.write(f"| {r['flight_id']} | {r['method']} | "
                     f"{r['fuel_kg']} | {r['duration_min']} | "
                     f"{r['cost']} | {'yes' if r['converged'] else 'no'} |\n")
    print(f"  markdown -> {md_path}")


if __name__ == "__main__":
    main()
