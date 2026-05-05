"""Polynomial wind-field tests."""
import numpy as np
import pytest

from src.common.io_config import load_wind_field
from src.common.wind_field import WindField, wind


def test_load_yaml() -> None:
    wf = load_wind_field()
    assert len(wf.Wx_coeffs) == 9
    assert len(wf.Wy_coeffs) == 9
    # PDF Table 2 first column: c0 of Wx is -21.151
    assert wf.Wx_coeffs[0] == pytest.approx(-21.151)
    assert wf.Wy_coeffs[0] == pytest.approx(-65.3035)


def test_constructor_validates_length() -> None:
    with pytest.raises(ValueError):
        WindField(Wx_coeffs=(1.0, 2.0), Wy_coeffs=(3.0, 4.0, 5.0))


def test_magnitudes_in_european_region() -> None:
    """Wind speeds at typical European points should be in the
    tens-of-m/s range, matching PDF Figure 1 colorbars."""
    wf = load_wind_field()
    for lam, phi in [(10.0, 50.0), (0.0, 45.0), (20.0, 50.0), (15.0, 55.0)]:
        Wx, Wy = wind(lam, phi, wf)
        assert -50.0 < Wx < 50.0, f"Wx out of expected band at ({lam},{phi}): {Wx}"
        assert -50.0 < Wy < 50.0, f"Wy out of expected band at ({lam},{phi}): {Wy}"


def test_array_inputs() -> None:
    wf = load_wind_field()
    lam = np.array([0.0, 10.0, 20.0])
    phi = np.array([45.0, 50.0, 55.0])
    Wx, Wy = wind(lam, phi, wf)
    assert Wx.shape == lam.shape
    assert Wy.shape == phi.shape
