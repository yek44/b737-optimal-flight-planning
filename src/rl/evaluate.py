"""Roll out the unified trained policy and persist :class:`FlightResult` outputs.

By default this evaluates the *single* policy at
``models/rl/unified/policy.zip`` on each of the three test flights, in
keeping with the project rule that one trained policy must fly all
scenarios. If the unified file is missing the loader falls back to the
legacy per-flight ``models/rl/flight_<id>/policy.zip`` layout so older
checkpoints stay loadable.

The output layout (``results/rl/flight_<id>/{solution.npz, meta.json}``
plus the 8 standard PNGs) is unchanged so the comparison overlays and
the report figures pick the files up without modification.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ..common.cost import TIME_PENALTY
from ..common.io_config import load_aircraft
from ..common.plotting import make_eight_plots
from ..common.results import FlightResult
from .env import EnvConfig, FlightPlannerEnv


def _locate_policy(models_root: Path) -> Path:
    """Find the policy to evaluate.

    Loads the unified single policy at ``models_root / "unified" / "policy.zip"``.
    """
    unified = models_root / "unified" / "policy.zip"
    if unified.exists():
        return unified
    raise FileNotFoundError(
        f"No unified policy.zip found under {models_root}/unified/."
    )


def evaluate(flight_id: int, models_root: Path, results_root: Path) -> FlightResult:
    """Roll the unified policy out on one test flight; persist results."""
    cfg = EnvConfig()
    policy_path = _locate_policy(models_root)
    model = PPO.load(policy_path, device="cpu")
    algo = "unified BC+DAgger"

    env = FlightPlannerEnv(flight_id=int(flight_id), cfg=cfg)
    obs, _ = env.reset(seed=0, options={"flight_id": int(flight_id)})
    done = False
    truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        action = action[0]
        obs, _, done, truncated, _ = env.step(action)

    t, state, control = env.trajectory_arrays()
    metrics = env.trajectory_metrics()
    if control.shape[0] == 0:
        control = np.zeros((1, 3))

    fp = env.fp
    fuel_total = metrics["fuel_final_kg"]
    duration = metrics["duration_final_s"]
    final_terminal_error = (
        (state[-1, 0] - env.dest_xyz[0]) ** 2
        + (state[-1, 1] - env.dest_xyz[1]) ** 2
        + (state[-1, 2] - env.dest_xyz[2]) ** 2
    )
    cost = fuel_total + TIME_PENALTY * duration + final_terminal_error

    fr = FlightResult(
        method="rl",
        flight_id=fp.id,
        t=t, state=state, control=control,
        fuel_total_kg=fuel_total,
        duration_s=duration,
        cost=float(cost),
        converged=bool(metrics["reached_goal"]),
        origin_lambda_deg=fp.initial.lambda_deg,
        origin_phi_deg=fp.initial.phi_deg,
        solver_meta={
            "algo": algo,
            "policy_path": str(policy_path),
            "is_unified": True,
            "dt_s": cfg.dt_s,
            "n_steps": int(state.shape[0]),
            "terminal_guidance": False,
            **metrics,
        },
    )
    outdir = results_root / f"flight_{flight_id}"
    fr.save(outdir)
    make_eight_plots(fr, load_aircraft(), outdir)
    print(fr.summary())
    print(
        f"  goal={metrics['reached_goal']} | "
        f"closest_xy={metrics['closest_xy_dist_m'] / 1000:.2f} km | "
        f"closest_goal_violation={metrics['closest_goal_violation_m']:.1f} m | "
        f"reason={metrics['termination_reason']}"
    )
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
