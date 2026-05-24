"""Generate a manifold of expert trajectories by perturbing initial conditions.

A single direct-collocation trajectory is not enough to train an imitation
policy that recovers from RK4 drift -- behavioural cloning on a one-shot
expert suffers from the classical *covariate shift* problem (states the
policy actually visits at evaluation time are not represented in the
training set). To fix this we solve the DC NLP from a small ball of
perturbed initial conditions around the nominal flight plan and collect
all the resulting (state, control) pairs. The aggregate dataset gives the
policy a "manifold of recovery actions" that covers the visited region.

Perturbations are drawn from a small uniform ball around the nominal
state:

    Delta lambda, phi:  +/- 0.5 deg
    Delta h:            +/- 200 m
    Delta v:            +/- 10 m/s
    Delta psi:          +/- 8 deg
    m unchanged (perturbing mass would mostly just rescale fuel)

For each perturbation we solve the DC NLP with the same destination,
warm-started from the nominal solution to keep solves cheap. Any
non-converged solve is discarded.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..common.io_config import FlightPlan, InitialState, get_flight_plan
from ..common.results import FlightResult
from ..direct.collocation import CollocationResult, solve_direct, to_flight_result
from ..common.io_config import load_aircraft, load_wind_field
from ..common.plotting import make_eight_plots


@dataclass(frozen=True)
class PerturbationConfig:
    """Bounds for the uniform perturbation of the initial state.

    All bounds are half-widths (the actual perturbation is sampled from
    [-x, +x] independently per component).
    """
    d_lambda_deg: float = 0.5
    d_phi_deg: float = 0.5
    d_h_m: float = 200.0
    d_v_mps: float = 10.0
    d_psi_deg: float = 8.0


def _perturb(fp: FlightPlan, rng: np.random.Generator,
             cfg: PerturbationConfig) -> FlightPlan:
    """Return a copy of ``fp`` with a uniformly perturbed initial state."""
    ini = fp.initial
    new_ini = InitialState(
        lambda_deg=ini.lambda_deg + rng.uniform(-cfg.d_lambda_deg, cfg.d_lambda_deg),
        phi_deg=ini.phi_deg + rng.uniform(-cfg.d_phi_deg, cfg.d_phi_deg),
        h_m=float(np.clip(ini.h_m + rng.uniform(-cfg.d_h_m, cfg.d_h_m),
                          1000.0, 12_500.0)),
        v_mps=float(np.clip(ini.v_mps + rng.uniform(-cfg.d_v_mps, cfg.d_v_mps),
                            125.0, 265.0)),
        psi_deg=ini.psi_deg + rng.uniform(-cfg.d_psi_deg, cfg.d_psi_deg),
        m_kg=ini.m_kg,
    )
    return dataclasses.replace(fp, initial=new_ini)


def generate_experts(
    flight_id: int,
    out_dir: Path,
    n_perturbations: int = 8,
    n_nodes: int = 41,
    seed: int = 0,
    cfg: Optional[PerturbationConfig] = None,
    warm_start: Optional[CollocationResult] = None,
    save_plots: bool = False,
) -> List[FlightResult]:
    """Generate the nominal + perturbed DC trajectories for one flight.

    Args:
        flight_id: Which flight (1, 2, or 3) to solve.
        out_dir: Directory to write the aggregated dataset and per-solve
            metadata.  Created if missing.
        n_perturbations: Number of perturbed solves on top of the nominal.
        n_nodes: Number of collocation nodes (default 41 matches the rest
            of the pipeline).
        seed: RNG seed for the perturbation draw.
        cfg: Perturbation bounds.  Defaults to :class:`PerturbationConfig`.
        warm_start: Optional warm-start result to reuse across solves.
        save_plots: If True, also dump the 8 standard plots for the
            nominal expert (useful for debugging).

    Returns:
        List of converged :class:`FlightResult` objects.  The first entry
        is always the nominal solve.
    """
    cfg = cfg or PerturbationConfig()
    rng = np.random.default_rng(seed)
    fp_nom = get_flight_plan(flight_id)
    ac = load_aircraft()
    wf = load_wind_field()

    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[FlightResult] = []

    print(f"[multi-expert] flight {flight_id}: solving nominal + "
          f"{n_perturbations} perturbations ...")

    # Nominal solve.
    cr_nom = solve_direct(fp_nom, ac, wf, n_nodes=n_nodes,
                          warm_start=warm_start)
    if cr_nom.converged:
        fr = to_flight_result(cr_nom, fp_nom, ac, n_nodes)
        results.append(fr)
        if save_plots:
            make_eight_plots(fr, ac, out_dir / "nominal")
    else:
        # If the nominal solve fails we cannot perturb usefully.
        raise RuntimeError(
            f"Nominal DC solve for flight {flight_id} did not converge "
            f"(status={cr_nom.return_status})"
        )

    # Perturbed solves -- warm-started from the nominal solution to keep
    # IPOPT iterations modest.
    for k in range(n_perturbations):
        fp_k = _perturb(fp_nom, rng, cfg)
        cr_k = solve_direct(fp_k, ac, wf, n_nodes=n_nodes,
                            warm_start=cr_nom)
        if cr_k.converged:
            fr_k = to_flight_result(cr_k, fp_k, ac, n_nodes)
            results.append(fr_k)
        else:
            print(f"  [multi-expert]   perturbation {k}: "
                  f"non-converged ({cr_k.return_status}); skipping")

    print(f"[multi-expert] flight {flight_id}: kept {len(results)} "
          f"experts (1 nominal + {len(results) - 1} perturbed)")
    return results


def aggregate_dataset(results: List[FlightResult]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (state, control) pairs from a list of experts.

    Each expert contributes ``N_i`` state nodes and ``N_i`` controls
    (the control at the final node is the held-final value already
    produced by :class:`CollocationResult`).  We drop the final pair so
    every entry is a genuine (state, action-taken-at-state) tuple.
    """
    states: List[np.ndarray] = []
    controls: List[np.ndarray] = []
    for fr in results:
        s = np.asarray(fr.state)
        u = np.asarray(fr.control)
        n = min(s.shape[0], u.shape[0]) - 1
        if n <= 0:
            continue
        states.append(s[:n])
        controls.append(u[:n])
    if not states:
        return np.zeros((0, 6)), np.zeros((0, 3))
    return np.concatenate(states, axis=0), np.concatenate(controls, axis=0)
