"""Behavioural cloning of the (multi-expert) direct-collocation policy.

Given a list of converged DC trajectories for one flight (typically the
nominal solve plus a handful of perturbed-initial-condition solves
produced by :mod:`src.rl.multi_expert`), this module:

1. Resamples each expert to the env's 10 s grid.
2. Replays each trajectory through :meth:`FlightPlannerEnv.observe_state`
   so the observations include the action history exactly as the live
   env would emit them.
3. Trains a Stable-Baselines3 PPO policy network (so the BC weights load
   cleanly into the on-policy fine-tuner) by minimising MSE between the
   policy's action mean and the expert action.
4. Runs the trained policy under several PyTorch RNG seeds and picks the
   seed whose deterministic rollout reaches the closest approach.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ..common.results import FlightResult
from .env import ACTION_HISTORY_LEN, EnvConfig, FlightPlannerEnv, config_for_flight
from .multi_expert import generate_experts


def _expert_to_pairs(fr: FlightResult, env: FlightPlannerEnv
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Convert a single expert trajectory to (obs, action) pairs.

    The expert is resampled onto the env's ``dt_s`` grid via piecewise
    linear interpolation in state and control. Observations are produced
    by :meth:`FlightPlannerEnv.observe_state`, so each pair carries the
    same action-history context the live env would supply.

    Note: when the perturbed expert starts at a non-zero ENU origin
    (because its flight plan moved the LLA origin), we shift it back so
    every expert is expressed in the *nominal* env's frame. This keeps
    the BC dataset consistent with what the policy actually sees at
    test time.
    """
    t_expert = np.asarray(fr.t)
    state_expert = np.asarray(fr.state).copy()
    control_expert = np.asarray(fr.control).copy()
    if control_expert.shape[0] == state_expert.shape[0] - 1:
        control_expert = np.vstack([control_expert, control_expert[-1:]])

    # Shift the expert into the nominal env's ENU frame.
    from ..common.coordinates import LLAOrigin, lla_to_enu
    nominal_origin = LLAOrigin(env.fp.initial.lambda_deg,
                               env.fp.initial.phi_deg)
    if (abs(fr.origin_lambda_deg - nominal_origin.lambda0_deg) > 1e-9
            or abs(fr.origin_phi_deg - nominal_origin.phi0_deg) > 1e-9):
        # Expert was solved from a shifted LLA origin.  Compute the offset
        # in nominal ENU frame and add it to every state.
        expert_origin = LLAOrigin(fr.origin_lambda_deg, fr.origin_phi_deg)
        x_off, y_off = lla_to_enu(expert_origin.lambda0_deg,
                                  expert_origin.phi0_deg,
                                  nominal_origin)
        state_expert[:, 0] += float(x_off)
        state_expert[:, 1] += float(y_off)

    dt = env.cfg.dt_s
    t_grid = np.arange(0.0, float(t_expert[-1]) + 1e-9, dt)

    state_resampled = np.stack([
        np.interp(t_grid, t_expert, state_expert[:, j])
        for j in range(6)
    ], axis=1)
    control_resampled = np.stack([
        np.interp(t_grid, t_expert, control_expert[:, j])
        for j in range(3)
    ], axis=1)

    # Normalise expert controls to [-1, 1] for the action space.
    a_arr = env.physical_to_action(control_resampled)

    # Reset once so the env's normalisation state (e.g. ``_init_dist``)
    # is initialised against the *nominal* IC.  We then poke states in
    # via ``observe_state`` without driving them through ``step``.
    env.reset(seed=0)

    obs_list: List[np.ndarray] = []
    n = state_resampled.shape[0]
    for k in range(n):
        # Action history at step k = actions taken at steps k-1, k-2, ...
        # The control at index k is what the policy emits AT step k, so
        # for the observation at step k the history is steps [k-3, k-2,
        # k-1] (clamped at zero).
        if k == 0:
            hist = None
        else:
            lo = max(0, k - ACTION_HISTORY_LEN)
            hist = a_arr[lo:k]
        obs_list.append(env.observe_state(state_resampled[k], k, hist))
    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
    return obs_arr, a_arr.astype(np.float32)


def build_dataset(experts: Sequence[FlightResult], flight_id: int,
                  cfg: Optional[EnvConfig] = None
                  ) -> tuple[np.ndarray, np.ndarray, FlightPlannerEnv]:
    """Stack (obs, action) pairs from every expert trajectory.

    Returns the concatenated arrays plus a fresh nominal env (handy for
    the caller to reuse without re-instantiating).
    """
    cfg = cfg or config_for_flight(flight_id)
    env = FlightPlannerEnv(flight_id=flight_id, cfg=cfg)
    obs_chunks: List[np.ndarray] = []
    act_chunks: List[np.ndarray] = []
    for fr in experts:
        o, a = _expert_to_pairs(fr, env)
        obs_chunks.append(o)
        act_chunks.append(a)
    if obs_chunks:
        obs = np.concatenate(obs_chunks, axis=0)
    else:
        obs = np.zeros((0, env.observation_space.shape[0]))
    act = np.concatenate(act_chunks, axis=0) if act_chunks else np.zeros((0, 3))
    return obs.astype(np.float32), act.astype(np.float32), env


def _make_ppo(flight_id: int, cfg: EnvConfig, seed: int) -> PPO:
    """Construct a fresh PPO model with the project's standard layout."""
    vec_env = DummyVecEnv([lambda: FlightPlannerEnv(flight_id=flight_id, cfg=cfg)])
    return PPO(
        "MlpPolicy", vec_env,
        learning_rate=3.0e-4,
        n_steps=512, batch_size=256, n_epochs=10,
        gamma=cfg.gamma, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        seed=seed, verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
        device="cpu",
    )


def train_bc_once(obs: np.ndarray, act: np.ndarray, flight_id: int,
                  cfg: EnvConfig, epochs: int, lr: float, batch_size: int,
                  seed: int) -> PPO:
    """Train a single BC policy under a given torch seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _make_ppo(flight_id, cfg, seed)
    policy = model.policy
    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    n = obs_t.shape[0]
    losses: List[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            obs_b = obs_t[mb]
            act_b = act_t[mb]
            dist = policy.get_distribution(obs_b)
            pred = dist.mode()
            loss = F.mse_loss(pred, act_b)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            epoch_loss += float(loss.detach()) * obs_b.shape[0]
        epoch_loss /= max(1, n)
        losses.append(epoch_loss)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  [BC seed={seed}] epoch {epoch + 1:4d}/{epochs}  "
                  f"loss={epoch_loss:.5f}")
    return model


def evaluate_rollout(model: PPO, flight_id: int,
                     cfg: EnvConfig) -> tuple[float, float]:
    """Roll a model out deterministically; return (closest_dist, fuel)."""
    env = FlightPlannerEnv(flight_id=flight_id, cfg=cfg)
    obs, _ = env.reset(seed=0)
    done = False
    trunc = False
    while not (done or trunc):
        a, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, _, done, trunc, _ = env.step(a[0])
    _, state, _ = env.trajectory_arrays()
    fuel = float(env.fp.initial.m_kg - state[-1, 5])
    return float(env._closest_dist), fuel


def _rollout_score(dist_m: float, fuel_kg: float,
                   disk_radius_m: float) -> float:
    """Lower is better. Reached-disk rollouts compared by fuel; unreached
    rollouts compared by closest-approach distance with a large offset so
    any reached rollout dominates any unreached one."""
    if dist_m <= disk_radius_m:
        return float(fuel_kg)
    return 1.0e9 + float(dist_m)


def bc_seed_search(
    flight_id: int,
    experts: Sequence[FlightResult],
    out_dir: Path,
    epochs: int = 600,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    seeds: Sequence[int] = (0, 7, 17, 31, 42, 100, 123, 1729, 2024),
) -> tuple[Path, dict]:
    """Train BC under several seeds; save & return the best rollout.

    Selection rubric: when any seed's deterministic rollout closes within
    the env's terminal disk, pick the one with the lowest fuel burn.
    Otherwise pick the one whose closest-approach distance is smallest.
    Picking by distance alone (the earlier behaviour) sometimes selected
    a seed that grazed the disk with much higher fuel; this rubric keeps
    the policy DC-competitive when several seeds reach the goal.
    """
    cfg = config_for_flight(flight_id)
    obs, act, _ = build_dataset(experts, flight_id, cfg=cfg)
    print(f"[BC] flight {flight_id}: {obs.shape[0]} (obs, action) pairs "
          f"from {len(experts)} expert trajectories")

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "bc_policy.zip"
    best_seed = seeds[0]
    best_score = float("inf")
    best_dist = float("inf")
    best_fuel = float("inf")
    metrics: dict = {"per_seed": []}
    for seed in seeds:
        model = train_bc_once(obs, act, flight_id, cfg, epochs=epochs,
                              lr=lr, batch_size=batch_size, seed=seed)
        cand_path = out_dir / f"bc_policy_seed{seed}.zip"
        model.save(cand_path)
        dist, fuel = evaluate_rollout(model, flight_id, cfg)
        score = _rollout_score(dist, fuel, cfg.terminal_radius_m)
        marker = "*best*" if score < best_score else ""
        print(f"  [BC seed={seed:4d}] closest={dist / 1000:.2f} km, "
              f"fuel={fuel:.0f} kg {marker}")
        metrics["per_seed"].append({"seed": seed, "closest_km": dist / 1000,
                                    "fuel_kg": fuel, "score": score})
        if score < best_score:
            best_score = score
            best_dist = dist
            best_fuel = fuel
            best_seed = seed
    winner = out_dir / f"bc_policy_seed{best_seed}.zip"
    shutil.copy(str(winner), str(best_path))
    print(f"[BC] flight {flight_id}: best seed={best_seed} "
          f"closest={best_dist / 1000:.2f} km, fuel={best_fuel:.0f} kg")
    metrics["best_seed"] = best_seed
    metrics["best_closest_km"] = best_dist / 1000
    metrics["best_fuel_kg"] = best_fuel
    metrics["best_score"] = best_score
    return best_path, metrics


def main() -> None:
    """Convenience CLI: generate experts -> train BC -> save policy."""
    p = argparse.ArgumentParser(
        description="Multi-expert BC pretraining for one flight."
    )
    p.add_argument("--flight", type=int, default=1)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--n-perturb", type=int, default=8)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--models-dir", type=str, default="models/rl")
    args = p.parse_args()

    flight_id = args.flight
    expert_dir = Path(args.results_dir) / "rl_experts" / f"flight_{flight_id}"
    expert_dir.mkdir(parents=True, exist_ok=True)

    experts = generate_experts(
        flight_id=flight_id,
        out_dir=expert_dir,
        n_perturbations=args.n_perturb,
    )
    out_dir = Path(args.models_dir) / f"flight_{flight_id}"
    bc_seed_search(flight_id, experts, out_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
