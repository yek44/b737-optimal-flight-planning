"""DAgger-style imitation refinement on top of multi-expert behavioural cloning.

After multi-expert BC the policy is competent on the expert manifold but
still drifts on long-horizon flights (Flight 3) or sharp-turn flights
(Flight 2) once it leaves the manifold. DAgger fixes the classical
covariate-shift problem by alternating:

    1. Roll out the current policy in the env, collecting visited states.
    2. For each visited state, query an "expert correction" -- here, the
       nearest neighbour in the aggregated expert dataset (by L2 in a
       state-scale-normalised 6-D embedding).
    3. Append the (visited_state, corrected_action) pairs to the BC
       dataset.
    4. Retrain the policy on the aggregated dataset.

Iterating this for a small number of rounds (3-5) collapses the rollout
drift without ever needing on-policy PPO. The nearest-neighbour expert
is much cheaper than re-solving the DC NLP from each visited state and,
empirically, gives almost the same improvement on the three test flights.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree
from stable_baselines3 import PPO

from ..common.results import FlightResult
from .bc_pretrain import build_dataset, evaluate_rollout, train_bc_once
from .env import EnvConfig, FlightPlannerEnv, config_for_flight
from .multi_expert import aggregate_dataset


# Scale used to normalise the 6-D state before nearest-neighbour search.
# Same magnitudes as ``FlightPlannerEnv._sx``, augmented with v and mass.
_STATE_SCALE = np.array([1.0e6, 1.0e6, 1.0e4, 250.0, 1.0, 1.0e5])


def _build_knn(experts: Sequence[FlightResult]) -> Tuple[cKDTree, np.ndarray]:
    """Index the aggregated expert (state, action) dataset for kNN queries.

    Returns:
        tree: A ``scipy.spatial.cKDTree`` over the scaled state embeddings.
        actions: The matching expert physical-control array (N, 3).
    """
    states_phys, controls = aggregate_dataset(list(experts))
    # Map psi (column 4) to (sin, cos) to avoid wrap-around artefacts in
    # the kNN distance.
    psi = states_phys[:, 4]
    states_emb = np.column_stack([
        states_phys[:, 0] / _STATE_SCALE[0],
        states_phys[:, 1] / _STATE_SCALE[1],
        states_phys[:, 2] / _STATE_SCALE[2],
        states_phys[:, 3] / _STATE_SCALE[3],
        np.sin(psi),
        np.cos(psi),
        states_phys[:, 5] / _STATE_SCALE[5],
    ])
    tree = cKDTree(states_emb)
    return tree, controls


def _embed_state(s: np.ndarray) -> np.ndarray:
    """Embed a 6-D state vector for kNN lookup (matches ``_build_knn``)."""
    psi = s[4]
    return np.array([
        s[0] / _STATE_SCALE[0],
        s[1] / _STATE_SCALE[1],
        s[2] / _STATE_SCALE[2],
        s[3] / _STATE_SCALE[3],
        float(np.sin(psi)),
        float(np.cos(psi)),
        s[5] / _STATE_SCALE[5],
    ])


def collect_rollout(model: PPO, flight_id: int,
                    cfg: EnvConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the policy out deterministically; return (states, actions, obs).

    ``actions`` are the raw policy outputs (the [-1, 1]^3 representation
    used by the BC trainer). ``obs`` are the observations the policy saw
    -- we re-use these directly in DAgger retraining so that the
    augmented BC dataset and the live env stay in lock-step on the
    action-history channels.
    """
    env = FlightPlannerEnv(flight_id=flight_id, cfg=cfg)
    obs, _ = env.reset(seed=0)
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    obs_list: List[np.ndarray] = []
    done = False
    trunc = False
    while not (done or trunc):
        a, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        states.append(env.state.copy())
        actions.append(np.asarray(a[0], dtype=np.float32))
        obs_list.append(obs.astype(np.float32))
        obs, _, done, trunc, _ = env.step(a[0])
    return (
        np.asarray(states, dtype=np.float64),
        np.asarray(actions, dtype=np.float32),
        np.asarray(obs_list, dtype=np.float32),
    )


def correct_with_knn(states: np.ndarray, tree: cKDTree,
                     expert_controls: np.ndarray,
                     env: FlightPlannerEnv,
                     residual_threshold: float = 0.05
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """For each visited state, return ``(keep_mask, corrected_actions)``.

    States that fall *inside* the expert manifold (kNN distance below
    ``residual_threshold`` in the scaled embedding) are not selected by
    ``keep_mask`` -- the BC policy already handles those; only the
    drifted states need correction.
    """
    if states.shape[0] == 0:
        return (np.zeros((0,), dtype=bool),
                np.zeros((0, 3), dtype=np.float32))
    emb = np.stack([_embed_state(s) for s in states], axis=0)
    dist, idx = tree.query(emb, k=1)
    nearest_u = expert_controls[idx]
    nearest_a = env.physical_to_action(nearest_u)
    keep = dist > residual_threshold
    return keep, nearest_a


def run_dagger(
    flight_id: int,
    experts: Sequence[FlightResult],
    bc_policy_path: Path,
    out_dir: Path,
    n_rounds: int = 4,
    epochs_per_round: int = 200,
    lr: float = 5.0e-4,
    batch_size: int = 128,
    seed: int = 0,
    residual_threshold: float = 0.05,
) -> Tuple[Path, dict]:
    """Iterative DAgger refinement starting from a BC checkpoint.

    Args:
        flight_id: Which scenario.
        experts: The aggregated DC expert trajectories.
        bc_policy_path: Path to the seed-selected BC policy zip.
        out_dir: Where to write the refined policy.
        n_rounds: Number of DAgger iterations.
        epochs_per_round: Optimisation epochs per round.
        lr: Adam learning rate.
        batch_size: SGD batch size.
        seed: RNG seed.
        residual_threshold: kNN distance below which a visited state is
            considered "on-manifold" and not added to the correction set.

    Returns:
        Path to the refined ``policy.zip`` plus a metrics dict.
    """
    cfg = config_for_flight(flight_id)
    obs_bc, act_bc, env = build_dataset(experts, flight_id, cfg=cfg)

    # Build kNN index over the expert (state, action) manifold.
    tree, expert_controls_phys = _build_knn(experts)
    print(f"[DAgger] flight {flight_id}: kNN index built over "
          f"{expert_controls_phys.shape[0]} expert pairs")

    # Load BC checkpoint and score it as round-0 baseline so we can
    # always fall back to the seed-search winner if every DAgger round
    # regresses.
    model = PPO.load(bc_policy_path, device="cpu")
    bc_dist, bc_fuel = evaluate_rollout(model, flight_id, cfg)
    print(f"[DAgger] flight {flight_id} BC baseline: "
          f"closest={bc_dist / 1000:.2f} km, fuel={bc_fuel:.0f} kg")

    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "policy.zip"
    # Seed best policy with the BC baseline -- DAgger only overwrites it
    # when a round strictly improves on the scoring rubric.
    model.save(policy_path)

    def _score(dist_m: float, fuel_kg: float) -> float:
        """Lower is better. Reached-disk rounds compared by fuel; unreached
        rounds compared by closest-approach distance with a stiff offset
        so any reached round dominates any unreached round."""
        disk_radius_m = cfg.terminal_radius_m
        if dist_m <= disk_radius_m:
            return float(fuel_kg)
        return 1.0e9 + float(dist_m)

    best_score = _score(bc_dist, bc_fuel)
    best_round = 0
    metrics: dict = {
        "rounds": [],
        "bc_baseline": {"closest_km": bc_dist / 1000, "fuel_kg": bc_fuel},
    }
    obs_aug = obs_bc.copy()
    act_aug = act_bc.copy()

    for r in range(n_rounds):
        states_r, actions_r, obs_r = collect_rollout(model, flight_id, cfg)
        # Get kNN-corrected expert actions for off-manifold states.
        keep, corr_actions = correct_with_knn(
            states_r, tree, expert_controls_phys, env,
            residual_threshold=residual_threshold,
        )
        n_kept = int(np.sum(keep))
        if n_kept > 0:
            obs_aug = np.concatenate([obs_aug, obs_r[keep]], axis=0)
            act_aug = np.concatenate([act_aug, corr_actions[keep]], axis=0)
        print(f"[DAgger] flight {flight_id} round {r + 1}/{n_rounds}: "
              f"rollout {states_r.shape[0]} steps, kept {n_kept} corrections "
              f"-> dataset {obs_aug.shape[0]} pairs")

        # Retrain the policy from scratch on the aggregated dataset for
        # a short burst.  Training from scratch each round (rather than
        # fine-tuning the current model in place) gives a cleaner
        # optimisation: we reset the optimiser state and avoid drift on
        # the value head, which we won't use here.
        model = train_bc_once(obs_aug, act_aug, flight_id, cfg,
                              epochs=epochs_per_round, lr=lr,
                              batch_size=batch_size, seed=seed)
        dist, fuel = evaluate_rollout(model, flight_id, cfg)
        round_score = _score(dist, fuel)
        improved = round_score < best_score
        marker = "*best*" if improved else ""
        print(f"[DAgger] round {r + 1}: closest={dist / 1000:.2f} km, "
              f"fuel={fuel:.0f} kg {marker}")
        metrics["rounds"].append({
            "round": r + 1,
            "kept": n_kept,
            "dataset_size": int(obs_aug.shape[0]),
            "closest_km": dist / 1000,
            "fuel_kg": fuel,
            "score": round_score,
        })
        if improved:
            best_score = round_score
            best_round = r + 1
            model.save(policy_path)

    metrics["best_round"] = best_round
    metrics["best_score"] = best_score
    print(f"[DAgger] flight {flight_id}: best round {best_round} "
          f"(score={best_score:.1f}) saved -> {policy_path}")
    return policy_path, metrics


def main() -> None:
    """CLI is intentionally stubbed; the orchestrator is the entrypoint."""
    p = argparse.ArgumentParser(description="DAgger refinement of a BC policy.")
    p.add_argument("--flight", type=int, default=1)
    p.parse_args()
    raise SystemExit(
        "DAgger CLI requires expert trajectories; use src.rl.pipeline instead."
    )


if __name__ == "__main__":
    main()
