"""Roll out the trained policy and persist a :class:`FlightResult`.

Loads ``models/rl/flight_<id>/policy.zip``, runs it deterministically in
the per-flight env, and writes the same canonical artefacts the direct-
collocation solver produces (``solution.npz``, ``meta.json``, the 8
standard PNG plots). Identical naming makes the cross-method overlay in
:mod:`src.compare.make_report_figs` trivial.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ..common.io_config import load_aircraft
from ..common.plotting import make_eight_plots
from ..common.results import FlightResult
from .env import FlightPlannerEnv, config_for_flight


def evaluate(flight_id: int, models_root: Path, results_root: Path) -> FlightResult:
    """Roll out one flight; persist results; return a ``FlightResult``."""
    cfg = config_for_flight(flight_id)
    indir = models_root / f"flight_{flight_id}"
    model = PPO.load(indir / "policy.zip", device="cpu")

    env = FlightPlannerEnv(flight_id=flight_id, cfg=cfg)
    obs, _ = env.reset(seed=0)
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, done, truncated, _ = env.step(action[0])

    t, state, control = env.trajectory_arrays()
    if control.shape[0] == 0:
        control = np.zeros((1, 3))

    fp = env.fp
    fuel_total = float(fp.initial.m_kg - state[-1, 5])
    duration = float(t[-1])
    cost = (cfg.time_weight * duration
            + (state[-1, 0] - env.dest_xyz[0]) ** 2
            + (state[-1, 1] - env.dest_xyz[1]) ** 2
            + (state[-1, 2] - env.dest_xyz[2]) ** 2)

    fr = FlightResult(
        method="rl",
        flight_id=fp.id,
        t=t, state=state, control=control,
        fuel_total_kg=fuel_total,
        duration_s=duration,
        cost=float(cost),
        converged=True,
        origin_lambda_deg=fp.initial.lambda_deg,
        origin_phi_deg=fp.initial.phi_deg,
        solver_meta={
            "algo": "BC+DAgger",
            "dt_s": cfg.dt_s,
            "n_steps": int(state.shape[0]),
            "closest_dist_m": float(env._closest_dist),
        },
    )
    outdir = results_root / f"flight_{flight_id}"
    fr.save(outdir)
    make_eight_plots(fr, load_aircraft(), outdir)
    print(fr.summary())
    print(f"  closest_dist: {env._closest_dist / 1000:.2f} km")
    return fr


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description="Evaluate the trained RL policy.")
    p.add_argument("--flight", type=str, default="all")
    p.add_argument("--models-dir", type=str, default="models/rl")
    p.add_argument("--results-dir", type=str, default="results/rl")
    args = p.parse_args()

    models_root = Path(args.models_dir)
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    if args.flight == "all":
        ids = [1, 2, 3]
    else:
        ids = [int(args.flight)]
    for fid in ids:
        evaluate(fid, models_root, results_root)


if __name__ == "__main__":
    main()
