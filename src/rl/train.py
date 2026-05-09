"""Train a PPO policy for one flight scenario.

Usage:
    python -m src.rl.train --flight 1 --steps 100000

Outputs (under `models/rl/flight_<id>/`):
    policy.zip         -- SB3 saved model
    vec_normalize.pkl  -- normalisation statistics
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import EnvConfig, FlightPlannerEnv


def make_env(flight_id: int, seed: int = 0):
    def _init():
        return FlightPlannerEnv(flight_id=flight_id, cfg=EnvConfig(), seed=seed)
    return _init


def train(flight_id: int, total_steps: int, models_root: Path,
          seed: int = 0, n_envs: int = 4) -> None:
    env_fns = [make_env(flight_id, seed + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)  # DummyVecEnv avoids Windows mp issues
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, gamma=0.995)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3.0e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
    )
    model.learn(total_timesteps=total_steps, progress_bar=False)

    outdir = models_root / f"flight_{flight_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    model.save(outdir / "policy.zip")
    vec_env.save(str(outdir / "vec_normalize.pkl"))
    print(f"[rl] Flight {flight_id}: saved policy to {outdir}/policy.zip")


def main() -> None:
    p = argparse.ArgumentParser(description="Train PPO for one flight.")
    p.add_argument("--flight", type=int, default=1)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--models-dir", type=str, default="models/rl")
    args = p.parse_args()
    train(args.flight, args.steps, Path(args.models_dir),
          seed=args.seed, n_envs=args.n_envs)


if __name__ == "__main__":
    main()
