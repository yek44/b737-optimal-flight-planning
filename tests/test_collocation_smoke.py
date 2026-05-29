"""End-to-end smoke test for the Hermite-Simpson direct-collocation NLP.

Solves a tiny version of Flight 1 (small ``n_nodes`` so the test runs in
a few seconds) and checks that the IPOPT/CasADi NLP returns a sensible
trajectory: the solve converges, the terminal state hits the destination
within the hard-equality tolerance, mass is monotone non-increasing, and
the control history stays inside the declared physical bounds. This
guards against regressions in the collocation builder (bound scaling,
defect formula, decision-vector layout) that the per-module unit tests
do not cover.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.io_config import get_flight_plan, load_aircraft, load_wind_field
from src.direct.collocation import solve_direct, to_flight_result


@pytest.fixture(scope="module")
def _f1_small_solution():
    """Solve F1 once at a coarse mesh; shared across the smoke checks."""
    fp = get_flight_plan(1)
    ac = load_aircraft()
    wf = load_wind_field()
    # 21 nodes is enough to confirm the NLP is well-posed but keeps the
    # test under ~5 s of wall time on a single core.
    cr = solve_direct(fp, ac, wf, n_nodes=21, warm_start=None, max_iter=1000)
    fr = to_flight_result(cr, fp, ac, n_nodes=21)
    return fp, ac, fr


def test_solver_converges(_f1_small_solution) -> None:
    """The NLP should converge (Solve_Succeeded or Solved_To_Acceptable_Level)."""
    _, _, fr = _f1_small_solution
    assert fr.converged, (
        f"Direct collocation failed to converge on F1 smoke test: "
        f"{fr.solver_meta}"
    )


def test_terminal_state_reaches_destination(_f1_small_solution) -> None:
    """Terminal (x, y, h) must satisfy the hard-equality terminal pin."""
    fp, _, fr = _f1_small_solution
    from src.common.coordinates import LLAOrigin, lla_to_enu
    origin = LLAOrigin(fp.initial.lambda_deg, fp.initial.phi_deg)
    xd, yd = lla_to_enu(fp.destination.lambda_deg, fp.destination.phi_deg, origin)
    final = fr.state[-1]
    # IPOPT terminal tolerance is 1e-6 in scaled units; the position
    # scales are 1e5 m and altitude 1e4 m, so allow ~1 m of slack.
    assert abs(final[0] - xd) < 5.0
    assert abs(final[1] - yd) < 5.0
    assert abs(final[2] - fp.destination.h_m) < 1.0


def test_mass_is_monotone_non_increasing(_f1_small_solution) -> None:
    """``dm/dt <= 0`` everywhere; mass cannot grow."""
    _, _, fr = _f1_small_solution
    mass = fr.state[:, 5]
    diffs = np.diff(mass)
    assert np.all(diffs <= 1e-3), (
        f"Found a mass increase of {float(diffs.max()):.3e} kg between two "
        f"nodes; mass must be monotone non-increasing."
    )


def test_state_bounds_respected(_f1_small_solution) -> None:
    """h, V, and m stay inside the declared physical bounds at every node."""
    _, ac, fr = _f1_small_solution
    h = fr.state[:, 2]
    v = fr.state[:, 3]
    m = fr.state[:, 5]
    assert h.min() >= ac.h_min_m - 1.0
    assert h.max() <= ac.h_max_m + 1.0
    assert v.min() >= ac.v_min_mps - 0.5
    assert v.max() <= ac.v_max_mps + 0.5
    assert m.min() >= ac.m_OEW_kg - 1.0


def test_control_bounds_respected(_f1_small_solution) -> None:
    """gamma, mu, delta stay inside their declared bounds at every node."""
    _, ac, fr = _f1_small_solution
    g = fr.control[:, 0]
    mu = fr.control[:, 1]
    d = fr.control[:, 2]
    deg = np.pi / 180.0
    # 1e-3 rad slack for IPOPT terminal tolerance.
    assert g.min() >= ac.gamma_min_deg * deg - 1e-3
    assert g.max() <= ac.gamma_max_deg * deg + 1e-3
    assert mu.min() >= ac.mu_min_deg * deg - 1e-3
    assert mu.max() <= ac.mu_max_deg * deg + 1e-3
    assert d.min() >= ac.delta_min - 1e-3
    assert d.max() <= ac.delta_max + 1e-3


def test_duration_and_fuel_positive(_f1_small_solution) -> None:
    """tf and fuel burn should both be positive."""
    _, _, fr = _f1_small_solution
    assert fr.duration_s > 0.0
    assert fr.fuel_total_kg > 0.0
