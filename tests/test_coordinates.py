"""ENU <-> LLA round-trip and great-circle helper tests."""
import numpy as np
import pytest

from src.common.coordinates import (
    LLAOrigin,
    enu_to_lla,
    haversine_distance,
    initial_bearing,
    lla_to_enu,
)


def test_enu_origin_is_zero() -> None:
    origin = LLAOrigin(lambda0_deg=10.0, phi0_deg=45.0)
    x, y = lla_to_enu(10.0, 45.0, origin)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_round_trip_under_1m_for_300km_offset() -> None:
    origin = LLAOrigin(lambda0_deg=10.0, phi0_deg=45.0)
    lam0, phi0 = 13.0, 47.0  # ~300 km away
    x, y = lla_to_enu(lam0, phi0, origin)
    lam1, phi1 = enu_to_lla(x, y, origin)
    assert lam1 == pytest.approx(lam0, abs=1e-9)
    assert phi1 == pytest.approx(phi0, abs=1e-9)


def test_haversine_known_distance() -> None:
    # London (51.5074N, -0.1278E) to Paris (48.8566N, 2.3522E) is ~344 km
    d = haversine_distance(-0.1278, 51.5074, 2.3522, 48.8566)
    assert d == pytest.approx(343_770.0, rel=0.02)


def test_initial_bearing_due_east() -> None:
    # From (0, 0) heading to (10, 0): bearing should be ~pi/2 (east).
    b = initial_bearing(0.0, 0.0, 10.0, 0.0)
    assert b == pytest.approx(np.pi / 2.0, abs=0.01)


def test_initial_bearing_due_north() -> None:
    b = initial_bearing(0.0, 0.0, 0.0, 10.0)
    assert b == pytest.approx(0.0, abs=1e-9)
