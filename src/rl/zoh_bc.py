"""Train a ZOH-compatible imitation policy from the direct solution.

The direct-collocation controls are defined on a sparse Hermite-Simpson
mesh. Replaying those controls as 10 s zero-order-hold actions in the
Gymnasium environment can drift noticeably, especially on Flight 2's
sharp initial turn. This module first builds an expert dataset that is
compatible with the RL environment: at each 10 s step it solves a small
bounded least-squares tracking problem for the constant control that
best moves the current simulated state toward the next direct state.

The resulting labels are still derived from the direct expert, but they
match the dynamics and action hold used by policy rollouts.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from ..common.results import FlightResult
from .bc_pretrain import train_bc_once
from .env import EnvConfig, FlightPlannerEnv, _rk4_step, config_for_flight


def _parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seed list."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _resample_direct(
    fr: FlightResult,
    cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a direct trajectory on the RL control grid."""
    t_grid = np.arange(0.0, float(fr.t[-1]) + 1.0e-9, cfg.dt_s)
    state = np.stack([
        np.interp(t_grid, fr.t, fr.state[:, j])
        for j in range(6)
    ], axis=1)

    control = np.asarray(fr.control)
    t_control = fr.t if control.shape[0] == fr.t.shape[0] else fr.t[:-1]
    action_ref = np.stack([
        np.interp(t_grid, t_control, control[:, j])
        for j in range(3)
    ], axis=1)
    return t_grid, state, action_ref


def _angle_residual(a: float, b: float) -> float:
    """Shortest signed angular residual ``a - b``."""
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _fit_step_control(
    env: FlightPlannerEnv,
    state: np.ndarray,
    target: np.ndarray,
    nominal_control: np.ndarray,
    previous_control: np.ndarray,
) -> np.ndarray:
    """Fit one held-constant control for the env's RK4 step."""
    lo = env._u_lo
    hi = env._u_hi
    nominal = np.clip(nominal_control, lo, hi)
    previous = np.clip(previous_control, lo, hi)

    state_scale = np.array([1000.0, 1000.0, 80.0, 2.0, 0.01, 50.0])
    control_scale = np.array([np.deg2rad(3.0), np.deg2rad(5.0), 0.2])

    def objective(control: np.ndarray) -> float:
        trial = _rk4_step(
            state,
            np.asarray(control, dtype=float),
            env.cfg.dt_s,
            env.ac,
            env.wf,
            env.origin,
        )
        residual = trial - target
        residual[4] = _angle_residual(trial[4], target[4])
        nominal_residual = (control - nominal) / control_scale
        rate_residual = (control - previous) / control_scale
        return float(
            np.sum((residual / state_scale) ** 2)
            + 0.005 * np.sum(nominal_residual ** 2)
            + 0.002 * np.sum(rate_residual ** 2)
        )

    guesses = [
        previous,
        nominal,
        np.array([0.0, 0.0, 0.7], dtype=float),
    ]
    best_control = nominal
    best_value = float("inf")
    for guess in guesses:
        result = minimize(
            objective,
            guess,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"maxiter": 80, "ftol": 1.0e-10},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_control = np.asarray(result.x, dtype=float)
    return np.clip(best_control, lo, hi)


def build_zoh_dataset(
    flight_id: int,
    direct_result: FlightResult,
    cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build ``(obs, action)`` labels by tracking the direct solution."""
    dataset_cfg = EnvConfig(**vars(cfg))
    dataset_cfg.terminal_radius_m = 0.0
    env = FlightPlannerEnv(flight_id=flight_id, cfg=dataset_cfg)
    t_grid, direct_state, direct_control = _resample_direct(direct_result, cfg)

    obs, _ = env.reset(seed=0)
    previous_control = direct_control[0]
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []

    for k in range(t_grid.shape[0] - 1):
        control = _fit_step_control(
            env=env,
            state=env.state.copy(),
            target=direct_state[k + 1],
            nominal_control=direct_control[k],
            previous_control=previous_control,
        )
        action = env.physical_to_action(control)
        obs_rows.append(obs.astype(np.float32))
        action_rows.append(action.astype(np.float32))
        previous_control = control
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break

    t_roll, state_roll, _ = env.trajectory_arrays()
    direct_interp = np.stack([
        np.interp(t_roll, direct_result.t, direct_result.state[:, j])
        for j in range(6)
    ], axis=1)
    tracking = {
        "n_pairs": len(obs_rows),
        "rollout_duration_s": float(t_roll[-1]),
        "xy_rmse_km": float(
            np.sqrt(np.mean(np.sum(
                (state_roll[:, :2] - direct_interp[:, :2]) ** 2,
                axis=1,
            ))) / 1000.0
        ),
        "h_rmse_m": float(np.sqrt(np.mean(
            (state_roll[:, 2] - direct_interp[:, 2]) ** 2
        ))),
        "v_rmse_mps": float(np.sqrt(np.mean(
            (state_roll[:, 3] - direct_interp[:, 3]) ** 2
        ))),
        "fuel_kg": float(env.fp.initial.m_kg - state_roll[-1, 5]),
    }
    return (
        np.asarray(obs_rows, dtype=np.float32),
        np.asarray(action_rows, dtype=np.float32),
        tracking,
    )


def _score_metrics(metrics: dict) -> float:
    """Lower-is-better rollout score for seed selection."""
    if not bool(metrics["reached_goal"]):
        return 1.0e9 + float(metrics["closest_goal_violation_m"])
    return (
        float(metrics["closest_xy_dist_m"])
        + 0.10 * abs(float(metrics["final_h_err_m"]))
        + 0.001 * float(metrics["fuel_final_kg"])
    )


def train_zoh_bc(
    flight_id: int,
    direct_result: FlightResult,
    out_dir: Path,
    seeds: Sequence[int],
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict:
    """Train seed candidates and save the best as ``policy.zip``."""
    cfg = config_for_flight(flight_id)
    obs, action, tracking = build_zoh_dataset(flight_id, direct_result, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("inf")
    best_seed = int(seeds[0])
    per_seed: list[dict] = []
    for seed in seeds:
        model = train_bc_once(
            obs,
            action,
            flight_id,
            cfg,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=int(seed),
        )
        candidate = out_dir / f"zoh_bc_policy_seed{int(seed)}.zip"
        model.save(candidate)

        from .bc_pretrain import evaluate_rollout_metrics
        rollout = evaluate_rollout_metrics(model, flight_id, cfg)
        score = _score_metrics(rollout)
        per_seed.append({
            "seed": int(seed),
            "goal_reached": bool(rollout["reached_goal"]),
            "closest_xy_km": float(rollout["closest_xy_dist_m"]) / 1000.0,
            "goal_miss_m": float(rollout["closest_goal_violation_m"]),
            "fuel_kg": float(rollout["fuel_final_kg"]),
            "duration_s": float(rollout["duration_final_s"]),
            "score": score,
        })
        print(
            f"[zoh_bc] seed={int(seed)} goal={rollout['reached_goal']} "
            f"closest={float(rollout['closest_xy_dist_m']) / 1000.0:.2f} km "
            f"fuel={float(rollout['fuel_final_kg']):.0f} kg score={score:.1f}"
        )
        if score < best_score:
            best_score = score
            best_seed = int(seed)

    best_path = out_dir / f"zoh_bc_policy_seed{best_seed}.zip"
    shutil.copy(str(best_path), str(out_dir / "zoh_bc_policy.zip"))
    shutil.copy(str(best_path), str(out_dir / "policy.zip"))
    summary = {
        "flight_id": flight_id,
        "source": "direct_zoh_tracker",
        "tracking_dataset": tracking,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "best_seed": best_seed,
        "best_score": best_score,
        "per_seed": per_seed,
    }
    with open(out_dir / "zoh_bc_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[zoh_bc] best seed={best_seed}; saved {out_dir / 'policy.zip'}")
    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train a ZOH-compatible BC policy from direct results."
    )
    parser.add_argument("--flight", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="0,42")
    parser.add_argument("--models-dir", type=str, default="models/rl")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    direct = FlightResult.load(
        Path(args.results_dir) / "direct" / f"flight_{args.flight}"
    )
    train_zoh_bc(
        flight_id=int(args.flight),
        direct_result=direct,
        out_dir=Path(args.models_dir) / f"flight_{args.flight}",
        seeds=_parse_seeds(args.seeds),
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
    )


if __name__ == "__main__":
    main()
