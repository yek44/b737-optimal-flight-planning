"""End-to-end training pipeline for the *single* unified RL policy.

The course requires one trained policy that flies all three test
scenarios -- not one policy per flight. This pipeline implements that
formulation:

    1. Multi-flight multi-expert dataset:
       :func:`generate_multi_flight_experts` solves the DC NLP for the
       nominal IC + a handful of perturbed ICs **for each of the three
       flights**, returning a single combined expert list.
    2. Mixed-flight BC seed search: :func:`bc_seed_search_unified`
       trains nine BC seeds on the mixed dataset, evaluates each on
       *every* test flight, and keeps the seed with the lowest
       aggregate :func:`_rollout_score` (sum across flights).
    3. Per-flight kNN DAgger: :func:`run_dagger_unified` rolls the
       policy out on each flight in turn, queries the matching
       per-flight kNN, and aggregates the corrections into a single
       retraining set. Best-of-rounds against the unified score.
    4. Persists exactly one ``models/rl/unified/policy.zip`` plus a
       summary JSON.

Run once:

    python -m src.rl.pipeline_unified
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .bc_pretrain import bc_seed_search_unified
from .dagger import run_dagger_unified
from .multi_expert import generate_multi_flight_experts
from .train import ppo_fine_tune_unified


def run_unified_pipeline(
    models_root: Path,
    results_root: Path,
    n_perturb: int = 6,
    bc_epochs: int = 400,
    dagger_rounds: int = 4,
    dagger_epochs: int = 120,
    ppo_steps: int = 600_000,
    bc_seeds: tuple[int, ...] = (0, 7, 17, 31, 42, 100, 123, 1729, 2024),
) -> Path:
    """Run the end-to-end unified-policy training pipeline.

    Args:
        models_root: Root of the per-method models directory; the unified
            policy lands at ``models_root/unified/policy.zip``.
        results_root: Root of the project results directory; multi-flight
            expert metadata is dumped at ``results_root/rl_experts/unified/``.
        n_perturb: Perturbed DC solves per flight (6 keeps the dataset
            around 21 experts).
        bc_epochs: BC epochs per seed.
        dagger_rounds: DAgger correction iterations.
        dagger_epochs: BC epochs per DAgger round.
        bc_seeds: PyTorch RNG seeds for the BC seed search.

    Returns:
        Path to the canonical ``policy.zip`` (one file for all flights).
    """
    out_dir = models_root / "unified"
    expert_dir = results_root / "rl_experts" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    t0 = time.time()

    # Stage 1: multi-flight multi-expert generation.
    experts = generate_multi_flight_experts(
        out_dir=expert_dir, n_perturbations=n_perturb,
    )
    summary["n_experts"] = len(experts)
    summary["experts_per_flight"] = {
        int(fid): sum(1 for fr in experts if fr.flight_id == fid)
        for fid in (1, 2, 3)
    }

    # Stage 2: BC seed search across mixed-flight data.
    bc_path, bc_metrics = bc_seed_search_unified(
        experts=experts, out_dir=out_dir,
        epochs=bc_epochs, seeds=bc_seeds,
    )
    summary["bc"] = bc_metrics

    # Stage 3: DAgger refinement, per-flight kNN, best-of-rounds.
    dagger_path, dagger_metrics = run_dagger_unified(
        experts=experts, bc_policy_path=bc_path, out_dir=out_dir,
        n_rounds=dagger_rounds, epochs_per_round=dagger_epochs,
    )
    summary["dagger"] = dagger_metrics

    # Stage 4: PPO on-policy fine-tuning starting from the best
    # imitation checkpoint. The unified policy then has two improvement
    # opportunities -- DAgger covariate-shift correction and on-policy
    # RL refinement -- and the best-of-rounds guard inside
    # ``ppo_fine_tune_unified`` falls back to the imitation baseline if
    # PPO regresses.
    policy_path, ppo_metrics = ppo_fine_tune_unified(
        bc_policy_path=dagger_path, experts=experts, out_dir=out_dir,
        total_steps=ppo_steps,
    )
    summary["ppo"] = ppo_metrics

    summary["wall_time_s"] = round(time.time() - t0, 1)
    with open(out_dir / "pipeline_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[pipeline-U] done in {summary['wall_time_s']}s")
    print(f"[pipeline-U] summary -> {out_dir / 'pipeline_summary.json'}")
    return policy_path


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description="Unified single-policy RL pipeline.")
    p.add_argument("--n-perturb", type=int, default=6)
    p.add_argument("--bc-epochs", type=int, default=400)
    p.add_argument("--dagger-rounds", type=int, default=4)
    p.add_argument("--dagger-epochs", type=int, default=120)
    p.add_argument("--ppo-steps", type=int, default=600_000)
    p.add_argument("--models-dir", type=str, default="models/rl")
    p.add_argument("--results-dir", type=str, default="results")
    args = p.parse_args()

    run_unified_pipeline(
        models_root=Path(args.models_dir),
        results_root=Path(args.results_dir),
        n_perturb=args.n_perturb,
        bc_epochs=args.bc_epochs,
        dagger_rounds=args.dagger_rounds,
        dagger_epochs=args.dagger_epochs,
        ppo_steps=args.ppo_steps,
    )


if __name__ == "__main__":
    main()
