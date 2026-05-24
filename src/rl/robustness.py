"""Out-of-training robustness benchmark for a trained policy.

Demonstrates that the BC + DAgger / PPO state-feedback policy actually
generalises -- it is **not** a hand-tuned replay of the open-loop direct
solution but a feedback controller that closes the loop on every state.
We test this by rolling the policy out from a basket of perturbed
initial conditions that were **not** in the multi-expert training set,
and reporting the disk-reach rate plus fuel statistics.

The perturbation distribution intentionally exceeds the multi-expert
generator's bounds slightly, so a fraction of the test ICs live outside
the BC training manifold; the disk-reach rate then quantifies
*extrapolation* rather than mere interpolation.

Usage:

    python -m src.rl.robustness --flight all --n-trials 20

Outputs:

* ``results/rl/flight_<id>/robustness.json`` -- per-trial metrics +
  summary statistics.
* console table with disk-reach rate, median fuel, fuel std.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from ..common.io_config import FlightPlan, InitialState, get_flight_plan
from .env import EnvConfig, FlightPlannerEnv, config_for_flight


def _perturbed_initial(fp: FlightPlan, rng: np.random.Generator,
                       sigma_lambda: float = 0.7,
                       sigma_phi: float = 0.7,
                       sigma_h: float = 300.0,
                       sigma_v: float = 12.0,
                       sigma_psi: float = 10.0) -> FlightPlan:
    """Sample one perturbed initial condition (Gaussian for extrapolation).

    Multi-expert training uses *uniform* perturbations with half-widths
    0.5 deg / 200 m / 10 m/s / 8 deg. We here use *Gaussian* draws with
    standard deviations slightly larger than those half-widths, so the
    test set occasionally pokes outside the training ball.
    """
    ini = fp.initial
    return dataclasses.replace(fp, initial=InitialState(
        lambda_deg=ini.lambda_deg + float(rng.normal(0.0, sigma_lambda)),
        phi_deg=ini.phi_deg + float(rng.normal(0.0, sigma_phi)),
        h_m=float(np.clip(ini.h_m + rng.normal(0.0, sigma_h),
                          1000.0, 12_500.0)),
        v_mps=float(np.clip(ini.v_mps + rng.normal(0.0, sigma_v),
                            130.0, 260.0)),
        psi_deg=ini.psi_deg + float(rng.normal(0.0, sigma_psi)),
        m_kg=ini.m_kg,
    ))


def _rollout_with_ic(model: PPO, flight_id: int, cfg: EnvConfig,
                     fp: FlightPlan) -> Dict[str, float]:
    """Roll the saved policy out from a custom initial state."""
    env = FlightPlannerEnv(flight_id=flight_id, cfg=cfg)
    env.fp = fp  # override the per-flight default
    # Recompute env's per-flight cached quantities derived from fp.
    from ..common.coordinates import LLAOrigin, lla_to_enu
    env.origin = LLAOrigin(fp.initial.lambda_deg, fp.initial.phi_deg)
    xd, yd = lla_to_enu(fp.destination.lambda_deg,
                        fp.destination.phi_deg, env.origin)
    env.dest_xyz = np.array([float(xd), float(yd), fp.destination.h_m])

    obs, _ = env.reset(seed=0)
    done = False
    trunc = False
    while not (done or trunc):
        a, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, done, trunc, _ = env.step(a[0])
    _, state, _ = env.trajectory_arrays()
    fuel = float(fp.initial.m_kg - state[-1, 5])
    return {
        "closest_dist_m": float(env._closest_dist),
        "fuel_kg": fuel,
        "reached_disk": bool(env._closest_dist <= cfg.terminal_radius_m),
        "duration_s": float(env._step_idx * cfg.dt_s),
    }


def benchmark_flight(flight_id: int, models_root: Path, results_root: Path,
                     n_trials: int = 20, seed: int = 12345) -> Dict:
    """Run the perturbed-IC benchmark for one flight."""
    cfg = config_for_flight(flight_id)
    fp_nom = get_flight_plan(flight_id)
    model = PPO.load(models_root / f"flight_{flight_id}" / "policy.zip",
                     device="cpu")

    rng = np.random.default_rng(seed)
    trials: List[Dict[str, float]] = []
    for k in range(n_trials):
        fp_k = _perturbed_initial(fp_nom, rng)
        result = _rollout_with_ic(model, flight_id, cfg, fp_k)
        result["trial"] = k
        result["initial_lambda_deg"] = fp_k.initial.lambda_deg
        result["initial_phi_deg"] = fp_k.initial.phi_deg
        result["initial_h_m"] = fp_k.initial.h_m
        result["initial_v_mps"] = fp_k.initial.v_mps
        result["initial_psi_deg"] = fp_k.initial.psi_deg
        trials.append(result)

    reached = [t for t in trials if t["reached_disk"]]
    summary = {
        "flight_id": flight_id,
        "n_trials": n_trials,
        "n_reached_disk": len(reached),
        "disk_reach_rate": len(reached) / n_trials,
        "median_closest_km": float(np.median(
            [t["closest_dist_m"] / 1000 for t in trials])),
        "median_fuel_kg": float(np.median([t["fuel_kg"] for t in trials])),
        "fuel_std_kg": float(np.std([t["fuel_kg"] for t in trials])),
    }
    if reached:
        summary["median_fuel_reached_kg"] = float(np.median(
            [t["fuel_kg"] for t in reached]))

    out_dir = results_root / f"flight_{flight_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "trials": trials}
    with open(out_dir / "robustness.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[robustness] flight {flight_id}: "
          f"{len(reached)}/{n_trials} reached disk "
          f"({100 * len(reached) / n_trials:.0f}%), "
          f"median fuel={summary['median_fuel_kg']:.0f} kg, "
          f"median closest={summary['median_closest_km']:.2f} km")
    return summary


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description="Out-of-training robustness benchmark.")
    p.add_argument("--flight", type=str, default="all")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--models-dir", type=str, default="models/rl")
    p.add_argument("--results-dir", type=str, default="results/rl")
    args = p.parse_args()
    ids = [1, 2, 3] if args.flight == "all" else [int(args.flight)]
    rows = []
    for fid in ids:
        rows.append(benchmark_flight(fid, Path(args.models_dir),
                                     Path(args.results_dir),
                                     n_trials=args.n_trials,
                                     seed=args.seed))
    print("\n=== Robustness summary (out-of-training perturbed ICs) ===")
    print(f"{'flight':>6} {'reached/trials':>15} "
          f"{'rate':>6} {'median_fuel':>14} {'med_closest':>12}")
    for r in rows:
        print(f"{r['flight_id']:>6} "
              f"{r['n_reached_disk']:>7}/{r['n_trials']:<6} "
              f"{r['disk_reach_rate'] * 100:>5.0f}% "
              f"{r['median_fuel_kg']:>11.0f} kg "
              f"{r['median_closest_km']:>9.2f} km")


if __name__ == "__main__":
    main()
