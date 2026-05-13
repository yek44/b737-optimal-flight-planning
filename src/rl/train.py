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
from typing import Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import EnvConfig, FlightPlannerEnv


def make_env(flight_id: int, seed: int = 0):
    def _init():
        return FlightPlannerEnv(flight_id=flight_id, cfg=EnvConfig(), seed=seed)
    return _init


def train(flight_id: int, total_steps: int, models_root: Path,
          seed: int = 0, n_envs: int = 4,
          bc_init: Optional[Path] = None,
          ent_coef: float = 0.0,
          learning_rate: float = 3.0e-4,
          skip_ppo: bool = False) -> None:
    """Train PPO policy.

    If ``skip_ppo`` is True and ``bc_init`` is provided, simply copies
    the BC-pretrained policy weights to ``policy.zip`` -- this is the
    pure-BC / imitation-learning mode, useful when the env reward
    landscape would destabilise on-policy PPO refinement.
    """
    if skip_ppo and bc_init is not None:
        outdir = models_root / f"flight_{flight_id}"
        outdir.mkdir(parents=True, exist_ok=True)
        # Just re-save the BC policy under the name PPO would use.
        bc_model = PPO.load(bc_init, device="cpu")
        bc_model.save(outdir / "policy.zip")
        print(f"[rl] Flight {flight_id}: BC policy adopted as final "
              f"(skip_ppo); saved -> {outdir}/policy.zip")
        return
    env_fns = [make_env(flight_id, seed + i) for i in range(n_envs)]
    # Train on RAW observations: the env already hand-normalises every
    # observation channel.  VecNormalize on top introduces extra stats
    # that drift relative to the BC-warm-started policy, so we skip it.
    vec_env = DummyVecEnv(env_fns)  # DummyVecEnv avoids Windows mp issues

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=learning_rate,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        # Must match the arch used in src.rl.bc_pretrain so the saved
        # BC weights load cleanly.
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
    )

    # Transfer BC-pretrained policy weights into the fresh PPO model.
    if bc_init is not None:
        bc_model = PPO.load(bc_init, device="cpu")
        # Copy MLP extractor + action net so PPO starts from BC mean.
        model.policy.load_state_dict(bc_model.policy.state_dict())
        # Shrink action std hard so PPO doesn't immediately erase the
        # BC mean.  -2.5 -> exp(-2.5) = 0.082; tight enough to preserve
        # the expert behaviour, broad enough for local refinement.
        with torch.no_grad():
            model.policy.log_std.fill_(-2.5)
        print(f"[rl] BC warm-start from {bc_init}")

    model.learn(total_timesteps=total_steps, progress_bar=False)

    outdir = models_root / f"flight_{flight_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    model.save(outdir / "policy.zip")
    print(f"[rl] Flight {flight_id}: saved policy to {outdir}/policy.zip")


def main() -> None:
    p = argparse.ArgumentParser(description="Train PPO for one flight.")
    p.add_argument("--flight", type=int, default=1)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--models-dir", type=str, default="models/rl")
    p.add_argument("--bc-init", type=str, default=None,
                   help="Path to BC-pretrained policy.zip to warm-start from.")
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3.0e-4)
    p.add_argument("--skip-ppo", action="store_true",
                   help="If set with --bc-init, save BC policy as final.")
    args = p.parse_args()
    bc = Path(args.bc_init) if args.bc_init else None
    train(args.flight, args.steps, Path(args.models_dir),
          seed=args.seed, n_envs=args.n_envs,
          bc_init=bc, ent_coef=args.ent_coef, learning_rate=args.lr,
          skip_ppo=args.skip_ppo)


if __name__ == "__main__":
    main()
