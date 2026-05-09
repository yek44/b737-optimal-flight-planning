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


def _solve_one(flight_id: int, n_nodes: int, results_root: Path,
               warm: Optional[CollocationResult]) -> CollocationResult:
    fp = get_flight_plan(flight_id)
    ac = load_aircraft()
    wf = load_wind_field()

    print(f"\n=== Flight {flight_id}: solving direct collocation "
          f"(n_nodes={n_nodes}) ===")
    t0 = time.time()
    cr = solve_direct(fp, ac, wf, n_nodes=n_nodes, warm_start=warm)
    elapsed = time.time() - t0

    fr = to_flight_result(cr, fp, ac, n_nodes)
    outdir = results_root / f"flight_{flight_id}"
    fr.save(outdir)
    make_eight_plots(fr, ac, outdir)

    print(fr.summary())
    print(f"  status={cr.return_status} | iters={cr.iter_count} "
          f"| wall={elapsed:.1f}s")
    return cr


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct collocation solver.")
    parser.add_argument("--flight", type=str, default="all",
                        help="Flight id (1, 2, 3) or 'all'.")
    parser.add_argument("--n-nodes", type=int, default=61,
                        help="Number of collocation nodes (default 61).")
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
