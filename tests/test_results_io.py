"""Round-trip persistence tests for :class:`FlightResult`.

Verifies that a result saved to disk via ``FlightResult.save`` round-trips
through ``FlightResult.load`` with bit-identical state/control arrays and
exactly preserved scalar metadata (including the nested ``solver_meta``
dict). This catches accidental dtype changes, NaN-corrupting
JSON serialisation, and missing fields on the loader side.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.common.results import FlightResult


def _make_dummy_result(method: str = "direct", flight_id: int = 1) -> FlightResult:
    """Build a small, deterministic FlightResult for I/O testing."""
    n = 11
    t = np.linspace(0.0, 1000.0, n)
    state = np.stack([
        np.linspace(0.0, 5.0e5, n),       # x_e [m]
        np.linspace(0.0, 3.0e5, n),       # y_n [m]
        np.linspace(8000.0, 9000.0, n),   # h   [m]
        np.full(n, 220.0),                # v   [m/s]
        np.full(n, 0.5),                  # psi [rad]
        np.linspace(71000.0, 68500.0, n)  # m   [kg]
    ], axis=1)
    control = np.stack([
        np.zeros(n),
        np.zeros(n),
        np.full(n, 0.5),
    ], axis=1)
    return FlightResult(
        method=method,
        flight_id=flight_id,
        t=t,
        state=state,
        control=control,
        fuel_total_kg=2500.0,
        duration_s=1000.0,
        cost=2515.0,
        converged=True,
        origin_lambda_deg=-4.0,
        origin_phi_deg=40.0,
        solver_meta={
            "scheme": "test",
            "n_nodes": int(n),
            "iter_count": 42,
            "return_status": "Solve_Succeeded",
            "nested": {"inner": [1, 2, 3]},
        },
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save() then load() reproduces every numeric and scalar field."""
    fr = _make_dummy_result()
    fr.save(tmp_path)
    loaded = FlightResult.load(tmp_path)

    np.testing.assert_array_equal(fr.t, loaded.t)
    np.testing.assert_array_equal(fr.state, loaded.state)
    np.testing.assert_array_equal(fr.control, loaded.control)

    assert loaded.method == fr.method
    assert loaded.flight_id == fr.flight_id
    assert loaded.fuel_total_kg == pytest.approx(fr.fuel_total_kg)
    assert loaded.duration_s == pytest.approx(fr.duration_s)
    assert loaded.cost == pytest.approx(fr.cost)
    assert loaded.converged == fr.converged
    assert loaded.origin_lambda_deg == pytest.approx(fr.origin_lambda_deg)
    assert loaded.origin_phi_deg == pytest.approx(fr.origin_phi_deg)
    assert loaded.solver_meta == fr.solver_meta


def test_save_creates_two_files(tmp_path: Path) -> None:
    """save() writes both solution.npz and meta.json under outdir."""
    fr = _make_dummy_result()
    fr.save(tmp_path)
    assert (tmp_path / "solution.npz").is_file()
    assert (tmp_path / "meta.json").is_file()


def test_meta_json_has_expected_keys(tmp_path: Path) -> None:
    """meta.json contains every documented scalar field."""
    fr = _make_dummy_result(method="rl", flight_id=2)
    fr.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    expected_keys = {
        "method", "flight_id", "fuel_total_kg", "duration_s", "cost",
        "converged", "origin_lambda_deg", "origin_phi_deg",
        "solver_meta", "n_states", "n_controls",
    }
    assert expected_keys.issubset(meta.keys())
    assert meta["method"] == "rl"
    assert meta["flight_id"] == 2
    assert meta["n_states"] == fr.state.shape[0]
    assert meta["n_controls"] == fr.control.shape[0]


def test_save_creates_outdir_if_missing(tmp_path: Path) -> None:
    """save() should mkdir(parents=True, exist_ok=True) on outdir."""
    nested = tmp_path / "a" / "b" / "c"
    fr = _make_dummy_result()
    fr.save(nested)
    assert (nested / "solution.npz").is_file()


def test_summary_is_one_line(tmp_path: Path) -> None:
    """summary() returns a single non-empty line containing method and id."""
    fr = _make_dummy_result(method="direct", flight_id=3)
    s = fr.summary()
    assert "\n" not in s
    assert "direct" in s
    assert "3" in s
