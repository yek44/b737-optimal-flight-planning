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
from .dagger import _build_knn, _embed_state, evaluate_rollout
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

    dist, fuel = evaluate_rollout(model, flight_id, cfg)
    print(f"[PPO] flight {flight_id}: closest={dist / 1000:.2f} km, "
          f"fuel={fuel:.0f} kg")
    return {
        "closest_km": dist / 1000,
        "fuel_kg": fuel,
        "total_steps": total_steps,
        "anchor_weight": anchor_weight,
    }


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
