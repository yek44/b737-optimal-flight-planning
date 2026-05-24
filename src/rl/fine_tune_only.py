"""Standalone PPO fine-tune entry point.

Re-runs only stages 1 (multi-expert generation, cheap because DC warm-
starts) and 4 (PPO fine-tune) of the pipeline, reusing the already-saved
BC policy at ``models/rl/flight_<id>/bc_policy.zip``. Useful when a
flight's BC baseline almost reaches the disk but you want PPO to nudge
it the last few kilometres without paying the cost of re-running the
9-seed BC search and 4 DAgger rounds.

Usage:

    python -m src.rl.fine_tune_only --flight 2 --ppo-steps 500000

The output is the canonical ``models/rl/flight_<id>/policy.zip`` (the
same artefact ``src.rl.evaluate`` consumes), keeping the rest of the
pipeline unchanged. A ``ppo_finetune_summary.json`` is dropped next to
it so the report can cite the exact wall-time and final closest/fuel
numbers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from stable_baselines3 import PPO

from .bc_pretrain import evaluate_rollout
from .env import config_for_flight
from .multi_expert import generate_experts
from .train import ppo_fine_tune


def _score(dist_m: float, fuel_kg: float, disk_radius_m: float) -> float:
    """Same rubric as bc_pretrain / dagger best-of selection."""
    if dist_m <= disk_radius_m:
        return float(fuel_kg)
    return 1.0e9 + float(dist_m)


def main() -> None:
    """CLI entrypoint -- intentionally minimal."""
    p = argparse.ArgumentParser(description="PPO fine-tune on an existing BC policy.")
    p.add_argument("--flight", type=int, default=2,
                   help="Flight id to fine-tune (1, 2, or 3).")
    p.add_argument("--ppo-steps", type=int, default=500_000,
                   help="Total env steps for PPO fine-tune.")
    p.add_argument("--ppo-lr", type=float, default=1.0e-4)
    p.add_argument("--anchor-weight", type=float, default=0.5)
    p.add_argument("--n-perturb", type=int, default=8,
                   help="Multi-expert perturbations (re-generated cheaply).")
    p.add_argument("--bc-source", type=str, default="bc_policy.zip",
                   help="Filename under models/rl/flight_<id>/ to warm-start from.")
    p.add_argument("--models-dir", type=str, default="models/rl")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    flight_id = int(args.flight)
    out_dir = Path(args.models_dir) / f"flight_{flight_id}"
    expert_dir = Path(args.results_dir) / "rl_experts" / f"flight_{flight_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_dir.mkdir(parents=True, exist_ok=True)

    bc_source = out_dir / args.bc_source
    if not bc_source.exists():
        raise SystemExit(f"BC source policy not found: {bc_source}")

    cfg = config_for_flight(flight_id)

    # Baseline: score the BC source so we can compare PPO to it later.
    bc_model = PPO.load(bc_source, device="cpu")
    bc_dist, bc_fuel = evaluate_rollout(bc_model, flight_id, cfg)
    bc_score = _score(bc_dist, bc_fuel, cfg.terminal_radius_m)
    print(f"[fine_tune_only] flight {flight_id}: BC baseline "
          f"closest={bc_dist / 1000:.2f} km, fuel={bc_fuel:.0f} kg "
          f"(score={bc_score:.1f})")

    print(f"[fine_tune_only] flight {flight_id}: regenerating experts "
          f"(n_perturb={args.n_perturb}) ...")
    t0 = time.time()
    experts = generate_experts(flight_id=flight_id, out_dir=expert_dir,
                               n_perturbations=args.n_perturb)
    print(f"[fine_tune_only] experts ready in {time.time() - t0:.1f}s "
          f"({len(experts)} converged)")

    # Stash the current policy.zip (likely == BC baseline) under a
    # backup name so we can restore it if PPO regresses.
    canonical = out_dir / "policy.zip"
    backup = out_dir / "policy_pre_ppo.zip"
    if canonical.exists():
        shutil.copy(str(canonical), str(backup))

    print(f"[fine_tune_only] PPO fine-tune from {bc_source.name} "
          f"({args.ppo_steps} steps, lr={args.ppo_lr}, "
          f"anchor={args.anchor_weight}) ...")
    t1 = time.time()
    metrics = ppo_fine_tune(
        flight_id=flight_id,
        policy_path=bc_source,
        experts=experts,
        out_dir=out_dir,
        total_steps=int(args.ppo_steps),
        learning_rate=float(args.ppo_lr),
        anchor_weight=float(args.anchor_weight),
    )
    metrics["wall_time_s"] = round(time.time() - t1, 1)
    metrics["bc_source"] = args.bc_source
    metrics["flight_id"] = flight_id
    metrics["bc_baseline"] = {
        "closest_km": bc_dist / 1000,
        "fuel_kg": bc_fuel,
        "score": bc_score,
    }
    ppo_score = _score(metrics["closest_km"] * 1000.0,
                       metrics["fuel_kg"], cfg.terminal_radius_m)
    metrics["ppo_score"] = ppo_score
    metrics["accepted"] = ppo_score < bc_score
    if not metrics["accepted"]:
        print(f"[fine_tune_only] PPO did not improve "
              f"(score {ppo_score:.1f} >= BC {bc_score:.1f}); "
              f"restoring BC baseline to policy.zip")
        if backup.exists():
            shutil.copy(str(backup), str(canonical))
    else:
        print(f"[fine_tune_only] PPO accepted "
              f"(score {ppo_score:.1f} < BC {bc_score:.1f})")

    summary_path = out_dir / "ppo_finetune_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[fine_tune_only] saved -> {summary_path}")
    print(f"[fine_tune_only] final: closest={metrics['closest_km']:.2f} km, "
          f"fuel={metrics['fuel_kg']:.0f} kg "
          f"(accepted={metrics['accepted']})")


if __name__ == "__main__":
    main()
