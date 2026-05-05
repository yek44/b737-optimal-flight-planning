"""Atmosphere model sanity tests."""
import numpy as np
import pytest

from src.common.atmosphere import rho
from src.common.constants import RHO0


def test_sea_level_density() -> None:
    assert rho(0.0) == pytest.approx(RHO0, rel=1e-9)


def test_density_decreases_with_altitude() -> None:
    h = np.linspace(0.0, 11_000.0, 23)
    r = np.array([rho(hi) for hi in h])
    assert np.all(np.diff(r) < 0.0), "rho must be strictly decreasing with h"


def test_density_at_11km_is_ballpark() -> None:
    # ISA gives ~0.364 kg/m^3 at the tropopause; the project formula
    # gives a close-but-not-identical value -- assert within 5%.
    r = float(rho(11_000.0))
    assert r == pytest.approx(0.364, rel=0.05)


def test_array_input() -> None:
    h = np.array([0.0, 5_000.0, 10_000.0])
    r = rho(h)
    assert r.shape == h.shape
    assert r[0] > r[1] > r[2]
