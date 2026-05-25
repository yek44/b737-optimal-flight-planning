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


# Flight-1 has the longest horizon (~13 000 s) and the worst observed
# out-of-training robustness on a Gaussian-IC test (15\% disk reach with
# the default uniform ball). Widening the F1 perturbation ball gives BC
# a denser recovery manifold for long-horizon track errors, which the
# policy needs because heading errors accumulate over time.
F1_PERTURBATION = PerturbationConfig(
    d_lambda_deg=1.0,
    d_phi_deg=1.0,
    d_h_m=300.0,
    d_v_mps=15.0,
    d_psi_deg=15.0,
)
# Flights 2 and 3 are short enough that the default 0.5 deg / 200 m /
# 10 m/s / 8 deg ball is already sufficient.
DEFAULT_PERTURBATION = PerturbationConfig()


def _flight_perturbation_cfg(flight_id: int) -> PerturbationConfig:
    """Per-flight perturbation config; F1 gets widened bounds."""
    return F1_PERTURBATION if flight_id == 1 else DEFAULT_PERTURBATION


def _flight_n_perturbations(flight_id: int, default_n: int) -> int:
    """Per-flight perturbation count; F1 gets extras for the longer ball."""
    # F1 ball is 8x larger in 5D volume than the default, so double the
    # sample count to keep coverage roughly comparable.
    return default_n * 2 if flight_id == 1 else default_n


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
    n_nodes: int = 241,
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
        n_nodes: Number of collocation nodes (default 121 matches the rest
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

    # Nominal solve. Try the caller-supplied warm start first if any,
    # then fall back to the great-circle cold guess if that fails (and
    # vice versa). F2 in particular needs the F1 warm start to converge
    # at n_nodes=121, while F1 itself needs the cold start.
    candidates: List[tuple[str, CollocationResult]] = []
    if warm_start is not None:
        candidates.append((
            "warm",
            solve_direct(fp_nom, ac, wf, n_nodes=n_nodes,
                         warm_start=warm_start),
        ))
    candidates.append((
        "cold",
        solve_direct(fp_nom, ac, wf, n_nodes=n_nodes, warm_start=None),
    ))
    converged = [c for c in candidates if c[1].converged]
    if not converged:
        statuses = ", ".join(f"{name}={cr.return_status}"
                             for name, cr in candidates)
        raise RuntimeError(
            f"Nominal DC solve for flight {flight_id} did not converge "
            f"({statuses})"
        )
    chosen_name, cr_nom = min(converged, key=lambda c: c[1].cost)
    print(f"  [multi-expert]   nominal selected: {chosen_name} "
          f"(cost={cr_nom.cost:.3e}, iters={cr_nom.iter_count})")
    fr = to_flight_result(cr_nom, fp_nom, ac, n_nodes)
    results.append(fr)
    if save_plots:
        make_eight_plots(fr, ac, out_dir / "nominal")

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


def generate_multi_flight_experts(
    out_dir: Path,
    n_perturbations: int = 6,
    n_nodes: int = 241,
    seed: int = 0,
    flight_ids: tuple[int, ...] = (1, 2, 3),
) -> List[FlightResult]:
    """Generate the multi-flight expert dataset for unified-policy BC.

    Solves nominal + ``n_perturbations`` perturbed DC NLPs for **each**
    flight, returning a flat list. The single policy will learn from
    this combined manifold so the same weights handle all scenarios.

    Per-flight RNG is offset so different flights see independent
    perturbations.

    Args:
        out_dir: Where to dump per-flight metadata.
        n_perturbations: Perturbed solves per flight (default 6 keeps
            the total dataset at 21 experts ~ tractable).
        n_nodes: Collocation node count (>= 121 per CLAUDE.md).
        seed: Base RNG seed; flight ``k`` uses ``seed + 100 * k``.
        flight_ids: Iterable of flight ids; usually all three.

    Returns:
        Flat list of converged ``FlightResult`` objects across all
        flights, in the order ``flight_ids`` were given.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_experts: List[FlightResult] = []
    # F1 typically converges cleanly from the great-circle guess; F2
    # needs F1's converged nominal as a warm start at n_nodes >= 121.
    # We pipe the first flight's nominal solution into every subsequent
    # flight's ``generate_experts`` call to maximise convergence
    # robustness.
    first_nominal_meta: Optional[CollocationResult] = None
    for k, fid in enumerate(flight_ids):
        per_flight_dir = out_dir / f"flight_{fid}"
        warm = first_nominal_meta
        fid_int = int(fid)
        sub = generate_experts(
            flight_id=fid_int,
            out_dir=per_flight_dir,
            n_perturbations=_flight_n_perturbations(fid_int, n_perturbations),
            n_nodes=n_nodes,
            seed=seed + 100 * k,
            cfg=_flight_perturbation_cfg(fid_int),
            warm_start=warm,
        )
        all_experts.extend(sub)
        if first_nominal_meta is None and sub:
            # Re-solve once to recover the CollocationResult (we only
            # kept the FlightResult). Cheap because cold convergence on
            # F1 already happened.
            from ..common.io_config import load_aircraft, load_wind_field
            fp = get_flight_plan(int(fid))
            first_nominal_meta = solve_direct(
                fp, load_aircraft(), load_wind_field(),
                n_nodes=n_nodes, warm_start=None,
            )
    print(f"[multi-expert] multi-flight: kept {len(all_experts)} experts "
          f"across {len(flight_ids)} flights")
    return all_experts


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
