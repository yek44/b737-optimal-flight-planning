"""End-to-end RL training pipeline for one flight.

This orchestrator runs the full BC + DAgger (+ optional PPO fine-tune)
sequence so a single command produces the canonical ``policy.zip`` and
metrics summary used by ``src.rl.evaluate``:

    1. Generate the multi-expert dataset by solving the DC NLP from
       the nominal initial state plus a handful of perturbed
       initial states (``src.rl.multi_expert``).
    2. Train a behavioural-cloning policy under several PyTorch seeds
       and select the best-rollout seed (``src.rl.bc_pretrain``).
    3. Refine the BC policy with kNN-correction DAgger
       (``src.rl.dagger``).
    4. (Optional) Run a short PPO fine-tune on top of the DAgger
       policy with an expert-anchor reward (``src.rl.train``).
    5. Persist policy + metrics under ``models/rl/flight_<id>/``.

Run once per flight:

    python -m src.rl.pipeline --flight 1
    python -m src.rl.pipeline --flight 2
    python -m src.rl.pipeline --flight 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .bc_pretrain import bc_seed_search
from .dagger import run_dagger
from .multi_expert import generate_experts
from .train import ppo_fine_tune


def run_pipeline(flight_id: int,
                 models_root: Path,
                 results_root: Path,
                 n_perturb: int = 8,
                 bc_epochs: int = 600,
                 dagger_rounds: int = 4,
                 dagger_epochs: int = 200,
                 ppo_steps: int = 0,
                 ppo_lr: float = 1.0e-4) -> Path:
    """Run the full BC+DAgger(+PPO) pipeline for one flight.

    Args:
        flight_id: Which flight (1, 2, or 3).
        models_root: ``models/rl/`` root; per-flight subdirs are created.
        results_root: ``results/`` root; per-flight expert metadata go
            under ``results/rl_experts/flight_<id>/``.
        n_perturb: Number of perturbed DC solves on top of the nominal.
        bc_epochs: Behavioural-cloning epochs per seed.
        dagger_rounds: Number of DAgger correction iterations.
        dagger_epochs: BC epochs per DAgger round.
        ppo_steps: If > 0, run a PPO fine-tune for this many env steps.
        ppo_lr: PPO learning rate.

    Returns:
        Path to the canonical ``policy.zip``.
    """
    out_dir = models_root / f"flight_{flight_id}"
    expert_dir = results_root / "rl_experts" / f"flight_{flight_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"flight_id": flight_id}
    t0 = time.time()

    # Stage 1: multi-expert generation.
    experts = generate_experts(flight_id=flight_id, out_dir=expert_dir,
                               n_perturbations=n_perturb)
    summary["n_experts"] = len(experts)
    summary["expert_fuel_kg"] = [round(fr.fuel_total_kg, 1) for fr in experts]

    # Stage 2: behavioural-cloning seed search.
    bc_path, bc_metrics = bc_seed_search(
        flight_id=flight_id, experts=experts, out_dir=out_dir,
        epochs=bc_epochs,
    )
    summary["bc"] = bc_metrics

    # Stage 3: DAgger refinement.
    policy_path, dagger_metrics = run_dagger(
        flight_id=flight_id, experts=experts, bc_policy_path=bc_path,
        out_dir=out_dir, n_rounds=dagger_rounds,
        epochs_per_round=dagger_epochs,
    )
    summary["dagger"] = dagger_metrics

    # Stage 4 (optional): PPO fine-tune with expert anchor.
    if ppo_steps > 0:
        ppo_metrics = ppo_fine_tune(
            flight_id=flight_id,
            policy_path=policy_path,
            experts=experts,
            out_dir=out_dir,
            total_steps=ppo_steps,
            learning_rate=ppo_lr,
        )
        summary["ppo"] = ppo_metrics

    summary["wall_time_s"] = round(time.time() - t0, 1)
    with open(out_dir / "pipeline_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[pipeline] flight {flight_id}: done in {summary['wall_time_s']}s")
    print(f"[pipeline] summary -> {out_dir / 'pipeline_summary.json'}")
    return policy_path


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description="Full BC+DAgger RL pipeline.")
    p.add_argument("--flight", type=str, default="all",
                   help="Flight id (1, 2, 3) or 'all'.")
    p.add_argument("--n-perturb", type=int, default=8)
    p.add_argument("--bc-epochs", type=int, default=600)
    p.add_argument("--dagger-rounds", type=int, default=4)
    p.add_argument("--dagger-epochs", type=int, default=200)
    p.add_argument("--ppo-steps", type=int, default=0)
    p.add_argument("--ppo-lr", type=float, default=1.0e-4)
    p.add_argument("--models-dir", type=str, default="models/rl")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    if args.flight == "all":
        ids = [1, 2, 3]
    else:
        ids = [int(args.flight)]

    models_root = Path(args.models_dir)
    results_root = Path(args.results_dir)
    for fid in ids:
        run_pipeline(
            flight_id=fid,
            models_root=models_root,
            results_root=results_root,
            n_perturb=args.n_perturb,
            bc_epochs=args.bc_epochs,
            dagger_rounds=args.dagger_rounds,
            dagger_epochs=args.dagger_epochs,
            ppo_steps=args.ppo_steps,
            ppo_lr=args.ppo_lr,
        )


if __name__ == "__main__":
    main()
