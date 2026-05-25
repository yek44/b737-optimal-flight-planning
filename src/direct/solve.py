"""CLI entrypoint: solve direct-collocation NLP and persist results.

Usage:
    python -m src.direct.solve --flight 1
    python -m src.direct.solve --flight all --warm-start

Saves `results/direct/flight_<id>/{solution.npz, meta.json}` and the
8 standard PNG plots into the same directory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from ..common.io_config import (
    get_flight_plan,
    load_aircraft,
    load_flight_plans,
    load_wind_field,
)
from ..common.plotting import make_eight_plots
from .collocation import CollocationResult, solve_direct, to_flight_result


def _solve_one(
    flight_id: int,
    n_nodes: int,
    results_root: Path,
    warm: Optional[CollocationResult],
) -> CollocationResult:
    fp = get_flight_plan(flight_id)
    ac = load_aircraft()
    wf = load_wind_field()

    print(f"\n=== Flight {flight_id}: solving direct collocation "
          f"(n_nodes={n_nodes}) ===")
    candidates: list[tuple[str, CollocationResult, float]] = []

    t0 = time.time()
    cold = solve_direct(fp, ac, wf, n_nodes=n_nodes, warm_start=None)
    candidates.append(("cold", cold, time.time() - t0))

    if warm is not None:
        t0 = time.time()
        warm_cr = solve_direct(fp, ac, wf, n_nodes=n_nodes, warm_start=warm)
        candidates.append(("warm_from_flight_1", warm_cr, time.time() - t0))

    converged = [c for c in candidates if c[1].converged]
    pool = converged if converged else candidates
    selected_name, cr, elapsed = min(pool, key=lambda c: c[1].cost)

    fr = to_flight_result(cr, fp, ac, n_nodes)
    fr.solver_meta["selected_initial_guess"] = selected_name
    fr.solver_meta["candidate_solutions"] = [
        {
            "initial_guess": name,
            "converged": cand.converged,
            "cost": cand.cost,
            "fuel_total_kg": float(fp.initial.m_kg - cand.state[-1, 5]),
            "duration_s": cand.tf,
            "iter_count": cand.iter_count,
            "return_status": cand.return_status,
            "wall_time_s": wall_s,
        }
        for name, cand, wall_s in candidates
    ]
    outdir = results_root / f"flight_{flight_id}"
    fr.save(outdir)
    make_eight_plots(fr, ac, outdir)

    print(fr.summary())
    for name, cand, wall_s in candidates:
        marker = "selected" if name == selected_name else "candidate"
        print(
            f"  {marker}: {name} | cost={cand.cost:.3e} | "
            f"fuel={fp.initial.m_kg - cand.state[-1, 5]:.1f} kg | "
            f"duration={cand.tf / 60.0:.1f} min | "
            f"status={cand.return_status} | iters={cand.iter_count} | "
            f"wall={wall_s:.1f}s"
        )
    print(f"  status={cr.return_status} | iters={cr.iter_count} "
          f"| wall={elapsed:.1f}s")
    return cr


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct collocation solver.")
    parser.add_argument("--flight", type=str, default="all",
                        help="Flight id (1, 2, 3) or 'all'.")
    parser.add_argument("--n-nodes", type=int, default=241,
                        help="Number of collocation nodes (default 241; "
                             "must be >= 121 per CLAUDE.md precision rule, "
                             "Option A precision tier uses 241).")
    parser.add_argument("--warm-start", action="store_true",
                        help="Reuse Flight 1's solution as guess for 2 and 3.")
    parser.add_argument("--results-dir", type=str, default="results/direct",
                        help="Output root directory.")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.flight == "all":
        ids = [fp.id for fp in load_flight_plans()]
    else:
        ids = [int(args.flight)]

    warm: Optional[CollocationResult] = None
    for fid in ids:
        cr = _solve_one(fid, args.n_nodes, results_root, warm)
        if args.warm_start and fid == 1:
            warm = cr  # reuse for subsequent flights


if __name__ == "__main__":
    main()
