"""Roll out a trained policy and persist a `FlightResult`."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..common.io_config import load_aircraft
from ..common.plotting import make_eight_plots
from ..common.results import FlightResult
from .env import EnvConfig, FlightPlannerEnv


def evaluate(flight_id: int, models_root: Path, results_root: Path) -> FlightResult:
    indir = models_root / f"flight_{flight_id}"
    model = PPO.load(indir / "policy.zip", device="cpu")

    # Re-create the env directly so we keep full trajectory history;
    # apply the saved VecNormalize obs statistics manually instead of
    # going through VecEnv (which auto-resets on done and clobbers
    # `inner.history_*`).
    inner = FlightPlannerEnv(flight_id=flight_id, cfg=EnvConfig())
    base_env = DummyVecEnv([lambda: inner])
    vn = VecNormalize.load(str(indir / "vec_normalize.pkl"), base_env)
    vn.training = False
    vn.norm_reward = False

    raw_obs, _ = inner.reset(seed=0)
    done = False
    truncated = False
    while not (done or truncated):
        norm_obs = vn.normalize_obs(raw_obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        raw_obs, _, done, truncated, _ = inner.step(action[0])

    t, state, control = inner.trajectory_arrays()
    if control.shape[0] == 0:
        # No steps executed -- skip; shouldn't happen in practice.
        control = np.zeros((1, 3))

    fp = inner.fp
    fuel_total = float(fp.initial.m_kg - state[-1, 5])
    duration = float(t[-1])
    cost = ((EnvConfig().time_weight + 0.0) * duration  # placeholder
            + (state[-1, 0] - inner.dest_xyz[0]) ** 2
            + (state[-1, 1] - inner.dest_xyz[1]) ** 2
            + (state[-1, 2] - inner.dest_xyz[2]) ** 2)

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
            "algo": "PPO",
            "dt_s": EnvConfig().dt_s,
            "n_steps": int(state.shape[0]),
        },
    )
    outdir = results_root / f"flight_{flight_id}"
    fr.save(outdir)
    make_eight_plots(fr, load_aircraft(), outdir)
    print(fr.summary())
    return fr


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate trained PPO policy.")
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
