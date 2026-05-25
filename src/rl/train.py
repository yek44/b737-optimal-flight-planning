"""PPO fine-tuning on top of a BC + DAgger policy.

Standalone PPO from a random init cannot solve the cruise-planning
problem at our compute budget, so this module is positioned only as a
*refinement* on top of an already-competent BC/DAgger checkpoint. The
fine-tuner reuses the standard env reward and optionally adds an
"expert anchor" bonus that mildly penalises straying from the nearest
expert state.

The most common usage is from :mod:`src.rl.pipeline`; this module also
exposes a thin CLI for ablation experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import torch
from scipy.spatial import cKDTree
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ..common.results import FlightResult
from .bc_pretrain import evaluate_rollout_metrics
from .dagger import _build_knn, _embed_state
from .env import EnvConfig, FlightPlannerEnv, config_for_flight


class ExpertAnchorWrapper(gym.Wrapper):
    """Adds a per-step expert-anchor bonus to the env reward.

    For each step the policy takes, we look up the nearest expert state
    in the aggregated DC dataset and apply

        r_anchor = - anchor_weight * d_emb^2

    where ``d_emb`` is the scaled L2 distance between the visited state
    and the nearest expert. The bonus is small enough that PPO is still
    driven by the original PDF reward, but it discourages the policy
    from running away from the expert manifold during fine-tuning.

    The wrapper accesses the wrapped env's true state via ``self.env.state``;
    that attribute is part of :class:`FlightPlannerEnv`'s public API.
    """

    def __init__(self, env: FlightPlannerEnv, tree: cKDTree,
                 anchor_weight: float = 0.5) -> None:
        super().__init__(env)
        self._tree = tree
        self._anchor_weight = float(anchor_weight)

    def step(self, action):  # type: ignore[override]
        obs, reward, term, trunc, info = self.env.step(action)
        s = self.env.unwrapped.state  # 6-D true state
        emb = _embed_state(s).reshape(1, -1)
        dist, _ = self._tree.query(emb, k=1)
        bonus = -self._anchor_weight * float(dist[0]) ** 2
        return obs, reward + bonus, term, trunc, info


def _make_anchored_env(flight_id: int, cfg: EnvConfig,
                       tree: cKDTree, anchor_weight: float):
    """Factory function for an env wrapped with the expert anchor."""
    def _init():
        base = FlightPlannerEnv(flight_id=flight_id, cfg=cfg)
        return ExpertAnchorWrapper(base, tree, anchor_weight=anchor_weight)
    return _init


def ppo_fine_tune(
    flight_id: int,
    policy_path: Path,
    experts: Sequence[FlightResult],
    out_dir: Path,
    total_steps: int = 500_000,
    learning_rate: float = 1.0e-4,
    anchor_weight: float = 0.5,
    n_envs: int = 4,
    log_std_init: float = -2.5,
) -> dict:
    """PPO refinement starting from a BC/DAgger checkpoint.

    Args:
        flight_id: Which flight (1, 2, or 3).
        policy_path: BC or DAgger ``policy.zip`` to warm-start from.
        experts: DC expert trajectories (used for the kNN anchor index).
        out_dir: Directory to overwrite ``policy.zip`` in.
        total_steps: Env steps to train for.
        learning_rate: PPO learning rate. Lower than 3e-4 so the BC mean
            isn't immediately erased.
        anchor_weight: Strength of the expert-anchor reward.
        n_envs: Number of parallel envs (DummyVecEnv on Windows).
        log_std_init: Initial log-stddev of the action distribution. -2.5
            keeps the BC mean essentially intact for the first few steps.

    Returns:
        Metrics dict with the final rollout closest-distance and fuel.
    """
    cfg = config_for_flight(flight_id)

    tree, _ = _build_knn(experts)
    env_fns = [
        _make_anchored_env(flight_id, cfg, tree, anchor_weight)
        for _ in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    bc_model = PPO.load(policy_path, device="cpu")
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=learning_rate,
        n_steps=512, batch_size=256, n_epochs=10,
        gamma=cfg.gamma, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
    )
    # Transfer BC/DAgger weights and shrink action std so PPO doesn't
    # instantly explore away from the warm start.
    model.policy.load_state_dict(bc_model.policy.state_dict())
    with torch.no_grad():
        model.policy.log_std.fill_(log_std_init)

    print(f"[PPO] flight {flight_id}: fine-tuning {total_steps} steps "
          f"from {policy_path.name}")
    model.learn(total_timesteps=total_steps, progress_bar=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "policy.zip")

    rollout = evaluate_rollout_metrics(model, flight_id, cfg)
    dist = float(rollout["closest_xy_dist_m"])
    fuel = float(rollout["fuel_final_kg"])
    print(f"[PPO] flight {flight_id}: goal={rollout['reached_goal']} "
          f"closest_xy={dist / 1000:.2f} km, "
          f"goal_miss={rollout['closest_goal_violation_m']:.0f} m, "
          f"fuel={fuel:.0f} kg")
    return {
        "goal_reached": rollout["reached_goal"],
        "closest_km": dist / 1000,
        "goal_miss_m": rollout["closest_goal_violation_m"],
        "fuel_kg": fuel,
        "total_steps": total_steps,
        "anchor_weight": anchor_weight,
    }


def _make_anchored_unified_env(cfg: EnvConfig, tree: cKDTree,
                               anchor_weight: float, seed: int):
    """Factory for an env that resamples one of F1/F2/F3 every reset()."""
    def _init():
        base = FlightPlannerEnv(flight_id="random", cfg=cfg, seed=seed)
        return ExpertAnchorWrapper(base, tree, anchor_weight=anchor_weight)
    return _init


def ppo_fine_tune_unified(
    bc_policy_path: Path,
    experts: Sequence[FlightResult],
    out_dir: Path,
    total_steps: int = 600_000,
    learning_rate: float = 5.0e-5,
    anchor_weight: float = 0.3,
    n_envs: int = 4,
    log_std_init: float = -2.5,
    seed: int = 17,
) -> tuple[Path, dict]:
    """PPO fine-tune the unified BC policy on a mixed-flight env.

    Continues training the BC-warm-started actor on a vectorized stack of
    envs that each draw a fresh flight id per ``reset()``. The expert
    anchor keeps the actor mean close to the DC manifold for the first
    few hundred thousand steps so PPO does not erase BC immediately.

    After training, the new policy is evaluated on each of the three test
    flights. If it reaches the disk on more flights or matches BC's
    disk-reach count with lower aggregate fuel, the PPO checkpoint is
    written to ``out_dir/policy.zip``; otherwise the BC baseline is kept.

    Args:
        bc_policy_path: Path to the unified BC ``policy.zip`` to warm
            start from. The PPO actor inherits this checkpoint's weights.
        experts: Multi-flight DC expert trajectories (used to build the
            kNN anchor index).
        out_dir: Models directory. The final accepted policy lands at
            ``out_dir / "policy.zip"``.
        total_steps: Total env steps for PPO. 600k is enough to refine BC
            on a 3-flight mixed env without overfitting.
        learning_rate: Lower than the SB3 default so the BC mean is not
            immediately overwritten. PPO inherently has a clipped policy
            update, but the value head starts cold and can swing wildly.
        anchor_weight: Per-step expert-anchor bonus magnitude. Set to 0.3
            so the PDF reward dominates while still discouraging drift.
        n_envs: Parallel envs (``DummyVecEnv`` for Windows-friendliness).
        log_std_init: Initial log-stddev. -2.5 keeps the BC mean
            essentially intact for the first few thousand steps.
        seed: PyTorch + env RNG seed for reproducibility.

    Returns:
        Tuple ``(policy_path, metrics)``. ``policy_path`` is always
        ``out_dir / "policy.zip"`` (either the PPO or BC checkpoint).
        ``metrics`` contains a ``per_flight`` mapping plus the aggregate
        BC/PPO scores and the ``accepted`` flag.
    """
    cfg = EnvConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree, _ = _build_knn(experts)
    env_fns = [
        _make_anchored_unified_env(cfg, tree, anchor_weight, seed + 7 * k)
        for k in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)

    torch.manual_seed(seed)
    bc_model = PPO.load(bc_policy_path, device="cpu")
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=learning_rate,
        n_steps=512, batch_size=256, n_epochs=10,
        gamma=cfg.gamma, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
        seed=seed,
    )
    model.policy.load_state_dict(bc_model.policy.state_dict())
    with torch.no_grad():
        model.policy.log_std.fill_(log_std_init)

    print(f"[PPO unified] fine-tuning {total_steps} steps from "
          f"{bc_policy_path.name}", flush=True)
    try:
        model.learn(total_timesteps=total_steps, progress_bar=False)
    except Exception as e:
        # Surface the failure with traceback so the pipeline log shows
        # why PPO bailed, then fall back to keeping the BC checkpoint as
        # the unified policy and continue downstream.
        import traceback
        print(f"[PPO unified] EXCEPTION during learn(): {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        bc_model.save(out_dir / "policy.zip")
        return out_dir / "policy.zip", {
            "total_steps": int(total_steps),
            "error": f"{type(e).__name__}: {e}",
            "accepted": "bc_due_to_ppo_crash",
        }

    ppo_tmp = out_dir / "ppo_candidate.zip"
    model.save(ppo_tmp)

    # Score BC and PPO on every flight; aggregate = sum(fuel) + 1e9 per missed disk.
    def _score(model_obj: PPO) -> tuple[dict, int, float]:
        per_flight = {}
        n_reached = 0
        agg = 0.0
        for fid in (1, 2, 3):
            r = evaluate_rollout_metrics(model_obj, fid, cfg)
            reached = bool(r["reached_goal"])
            fuel = float(r["fuel_final_kg"])
            per_flight[str(fid)] = {
                "reached_goal": reached,
                "fuel_kg": fuel,
                "closest_xy_km": float(r["closest_xy_dist_m"]) / 1000.0,
            }
            n_reached += int(reached)
            agg += fuel + (0.0 if reached else 1.0e9)
        return per_flight, n_reached, agg

    bc_pf, bc_n, bc_agg = _score(bc_model)
    ppo_pf, ppo_n, ppo_agg = _score(model)

    accept_ppo = (ppo_n > bc_n) or (ppo_n == bc_n and ppo_agg < bc_agg)

    if accept_ppo:
        model.save(out_dir / "policy.zip")
        chosen = "ppo"
    else:
        bc_model.save(out_dir / "policy.zip")
        chosen = "bc"

    print(f"[PPO unified] chosen: {chosen} | bc n_reached={bc_n} agg={bc_agg:.3e} | "
          f"ppo n_reached={ppo_n} agg={ppo_agg:.3e}")

    metrics = {
        "total_steps": int(total_steps),
        "learning_rate": float(learning_rate),
        "anchor_weight": float(anchor_weight),
        "bc": {"n_reached": bc_n, "score": bc_agg, "per_flight": bc_pf},
        "ppo": {"n_reached": ppo_n, "score": ppo_agg, "per_flight": ppo_pf},
        "accepted": chosen,
    }
    return out_dir / "policy.zip", metrics


def main() -> None:
    """CLI is intentionally stubbed; the orchestrator is the entrypoint."""
    p = argparse.ArgumentParser(description="PPO fine-tune from BC/DAgger.")
    p.add_argument("--flight", type=int, default=1)
    p.parse_args()
    raise SystemExit(
        "Standalone PPO fine-tune requires the expert dataset; "
        "use src.rl.pipeline --ppo-steps <N> instead."
    )


if __name__ == "__main__":
    main()
