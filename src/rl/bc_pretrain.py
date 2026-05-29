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
from .env import (
    ACTION_HISTORY_LEN,
    FLIGHT_IDS,
    OBS_DIM,
    EnvConfig,
    FlightPlannerEnv,
    config_for_flight,
)


def _expert_to_pairs(fr: FlightResult, env: FlightPlannerEnv
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Convert a single expert trajectory to (obs, action) pairs.

    The expert is resampled onto the env's ``dt_s`` grid via piecewise
    linear interpolation in state and control. The env is rebound to
    ``fr.flight_id`` before observations are computed so that the
    one-hot flight encoding and the per-flight destination embedded in
    each observation match the expert's actual scenario. After the call
    the env is left bound to whichever flight it was on.

    When the perturbed expert was solved from a shifted LLA origin we
    transform every (x, y) sample back to the flight's nominal ENU
    frame so the BC dataset is consistent with what the live policy
    sees when ``reset`` puts the aircraft at ``(0, 0)`` of the nominal
    origin.
    """
    # Bind the env to this expert's flight; per-flight destination /
    # one-hot follow automatically through the rebind.
    env._bind_flight(fr.flight_id)

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

    # Reset against this expert's own flight so ``_init_dist`` reflects
    # the right scenario for obs normalisation. We then poke states in
    # via ``observe_state`` without driving them through ``step``.
    env.reset(seed=0, options={"flight_id": fr.flight_id})

    obs_list: List[np.ndarray] = []
    a_list: List[np.ndarray] = []
    n = state_resampled.shape[0]
    step_stride = max(1, int(1.0 / dt))
    for k in range(0, n, step_stride):
        # Action history at decision step k contains the decisions at k-1, k-2, ...
        if k == 0:
            hist = None
        else:
            lo = max(0, k - ACTION_HISTORY_LEN * step_stride)
            hist = a_arr[lo:k:step_stride]
        obs_list.append(env.observe_state(state_resampled[k], k, hist))
        a_list.append(a_arr[k])
    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
    a_arr = np.stack(a_list, axis=0).astype(np.float32)
    return obs_arr, a_arr


def build_dataset(experts: Sequence[FlightResult],
                  flight_id: int | str = "random",
                  cfg: Optional[EnvConfig] = None
                  ) -> tuple[np.ndarray, np.ndarray, FlightPlannerEnv]:
    """Stack (obs, action) pairs from every expert trajectory.

    Handles mixed-flight expert lists transparently: each expert's
    obs is computed with the env bound to its own ``flight_id`` (see
    :func:`_expert_to_pairs`). The returned env is left in
    ``flight_id`` mode for the caller's convenience -- pass
    ``flight_id="random"`` to let downstream training pick a fresh
    flight per reset.
    """
    cfg = cfg or EnvConfig()
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
        obs = np.zeros((0, OBS_DIM), dtype=np.float32)
    act = np.concatenate(act_chunks, axis=0) if act_chunks else np.zeros((0, 3))
    return obs.astype(np.float32), act.astype(np.float32), env


def _make_ppo(flight_id: int | str, cfg: EnvConfig, seed: int) -> PPO:
    """Construct a fresh PPO model with the project's standard layout.

    The network is wider (``[256, 256]``) than the per-flight default
    because the unified policy has to specialise to three scenarios
    via the flight one-hot, and the previous ``[128, 128]`` MLP was
    visibly capacity-bound when trained on the mixed dataset.
    """
    vec_env = DummyVecEnv([lambda: FlightPlannerEnv(flight_id=flight_id, cfg=cfg)])
    return PPO(
        "MlpPolicy", vec_env,
        learning_rate=3.0e-4,
        n_steps=512, batch_size=256, n_epochs=10,
        gamma=cfg.gamma, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        seed=seed, verbose=0,
        policy_kwargs={"net_arch": [512, 512]},
        device="cpu",
    )


def train_bc_once(obs: np.ndarray, act: np.ndarray, flight_id: int | str,
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


def evaluate_rollout_metrics(
    model: PPO,
    flight_id: int,
    cfg: EnvConfig,
) -> dict:
    """Roll a model out deterministically on one flight; return goal metrics."""
    env = FlightPlannerEnv(flight_id=int(flight_id), cfg=cfg)
    obs, _ = env.reset(seed=0)
    done = False
    trunc = False
    while not (done or trunc):
        a, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        a = a[0]
        obs, _, done, trunc, _ = env.step(a)
    return env.trajectory_metrics()


def evaluate_rollout(model: PPO, flight_id: int,
                     cfg: EnvConfig) -> tuple[float, float]:
    """Roll a model out deterministically; return legacy (closest_xy, fuel)."""
    metrics = evaluate_rollout_metrics(model, flight_id, cfg)
    return float(metrics["closest_xy_dist_m"]), float(metrics["fuel_final_kg"])


def _rollout_score(metrics: dict) -> float:
    """Lower is better; successful rollouts compare by fuel, failures by miss."""
    if bool(metrics["reached_goal"]):
        return float(metrics["fuel_final_kg"])
    return 1.0e9 + float(metrics["closest_goal_violation_m"])


def evaluate_unified_rollout(
    model: PPO,
    cfg: EnvConfig,
    flight_ids: Sequence[int] = FLIGHT_IDS,
) -> dict:
    """Run one policy on every test flight and return per-flight + aggregate metrics.

    The aggregate score is the sum of per-flight scores under
    :func:`_rollout_score`, so a single bad flight cannot mask two good
    ones. This is the criterion :func:`bc_seed_search_unified` uses to
    pick the best seed.
    """
    per_flight: dict[int, dict] = {}
    total_score = 0.0
    reached = 0
    for fid in flight_ids:
        m = evaluate_rollout_metrics(model, int(fid), cfg)
        per_flight[int(fid)] = m
        total_score += _rollout_score(m)
        if m["reached_goal"]:
            reached += 1
    return {
        "per_flight": per_flight,
        "total_score": total_score,
        "n_reached": reached,
        "n_flights": len(flight_ids),
    }


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
    best_goal = False
    best_goal_miss = float("inf")
    metrics: dict = {"per_seed": []}
    for seed in seeds:
        model = train_bc_once(obs, act, flight_id, cfg, epochs=epochs,
                              lr=lr, batch_size=batch_size, seed=seed)
        cand_path = out_dir / f"bc_policy_seed{seed}.zip"
        model.save(cand_path)
        rollout = evaluate_rollout_metrics(model, flight_id, cfg)
        dist = float(rollout["closest_xy_dist_m"])
        fuel = float(rollout["fuel_final_kg"])
        score = _rollout_score(rollout)
        marker = "*best*" if score < best_score else ""
        print(f"  [BC seed={seed:4d}] goal={rollout['reached_goal']} "
              f"closest_xy={dist / 1000:.2f} km, "
              f"goal_miss={rollout['closest_goal_violation_m']:.0f} m, "
              f"fuel={fuel:.0f} kg {marker}")
        metrics["per_seed"].append({"seed": seed, "closest_km": dist / 1000,
                                    "goal_reached": rollout["reached_goal"],
                                    "goal_miss_m": rollout["closest_goal_violation_m"],
                                    "fuel_kg": fuel, "score": score})
        if score < best_score:
            best_score = score
            best_dist = dist
            best_fuel = fuel
            best_goal = bool(rollout["reached_goal"])
            best_goal_miss = float(rollout["closest_goal_violation_m"])
            best_seed = seed
    winner = out_dir / f"bc_policy_seed{best_seed}.zip"
    shutil.copy(str(winner), str(best_path))
    print(f"[BC] flight {flight_id}: best seed={best_seed} "
          f"goal={best_goal} closest_xy={best_dist / 1000:.2f} km, "
          f"goal_miss={best_goal_miss:.0f} m, fuel={best_fuel:.0f} kg")
    metrics["best_seed"] = best_seed
    metrics["best_goal_reached"] = best_goal
    metrics["best_closest_km"] = best_dist / 1000
    metrics["best_goal_miss_m"] = best_goal_miss
    metrics["best_fuel_kg"] = best_fuel
    metrics["best_score"] = best_score
    return best_path, metrics


def bc_seed_search_unified(
    experts: Sequence[FlightResult],
    out_dir: Path,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1.0e-3,
    seeds: Sequence[int] = (0, 7, 17, 31, 42, 100, 123, 1729, 2024),
    flight_ids: Sequence[int] = FLIGHT_IDS,
) -> tuple[Path, dict]:
    """BC seed search for the unified (single-policy) formulation.

    Trains one PPO network per seed on the *mixed-flight* expert
    dataset, evaluates it on every flight individually, and keeps the
    seed whose aggregate :func:`_rollout_score` (sum across flights) is
    smallest. This matches the project rule that a single policy must
    fly all three scenarios.
    """
    cfg = EnvConfig()
    obs, act, _ = build_dataset(experts, flight_id="random", cfg=cfg)
    print(f"[BC unified] {obs.shape[0]} (obs, action) pairs from "
          f"{len(experts)} experts across "
          f"{len({fr.flight_id for fr in experts})} flights")

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "bc_policy.zip"
    best_seed = seeds[0]
    best_score = float("inf")
    best_summary: dict = {}
    metrics: dict = {"per_seed": []}
    for seed in seeds:
        model = train_bc_once(obs, act, "random", cfg, epochs=epochs,
                              lr=lr, batch_size=batch_size, seed=seed)
        cand_path = out_dir / f"bc_policy_seed{seed}.zip"
        model.save(cand_path)
        agg = evaluate_unified_rollout(model, cfg, flight_ids=flight_ids)
        score = float(agg["total_score"])
        marker = "*best*" if score < best_score else ""
        per = agg["per_flight"]
        print(f"  [BC seed={seed:4d}] reached={agg['n_reached']}/"
              f"{agg['n_flights']} | "
              + ", ".join([
                  f"F{fid}: miss={per[fid]['closest_goal_violation_m']:.0f}m, "
                  f"fuel={per[fid]['fuel_final_kg']:.0f}kg"
                  for fid in flight_ids
              ]) + f"  score={score:.3e} {marker}")
        per_seed_row: dict[str, object] = {
            "seed": int(seed),
            "score": score,
            "n_reached": int(agg["n_reached"]),
            "per_flight": {
                int(fid): {
                    "reached_goal": bool(per[fid]["reached_goal"]),
                    "goal_miss_m": float(per[fid]["closest_goal_violation_m"]),
                    "fuel_kg": float(per[fid]["fuel_final_kg"]),
                    "duration_s": float(per[fid]["duration_reported_s"]),
                }
                for fid in flight_ids
            },
        }
        metrics["per_seed"].append(per_seed_row)
        if score < best_score:
            best_score = score
            best_seed = seed
            best_summary = per_seed_row
    winner = out_dir / f"bc_policy_seed{best_seed}.zip"
    shutil.copy(str(winner), str(best_path))
    print(f"[BC unified] best seed={best_seed}, "
          f"reached {best_summary['n_reached']}/{len(flight_ids)} disks, "
          f"score={best_score:.3e}")
    metrics["best_seed"] = int(best_seed)
    metrics["best_score"] = float(best_score)
    metrics["best_summary"] = best_summary
    return best_path, metrics


def main() -> None:
    """CLI is intentionally stubbed; the orchestrator is the entrypoint."""
    p = argparse.ArgumentParser(
        description="Multi-expert BC pretraining."
    )
    p.parse_args()
    raise SystemExit(
        "BC pretraining CLI is stubbed; use src.rl.pipeline_unified instead."
    )


if __name__ == "__main__":
    main()
