"""6-state dynamics sanity tests.

These do not depend on CasADi; they exercise the numpy backend only.

Heading-angle convention (NON-AVIATION standard):
    The PDF gives dx = v*cos(psi)*cos(gamma), dy = v*sin(psi)*cos(gamma).
    With (x, y) = (east, north), this means psi is measured COUNTER-
    CLOCKWISE FROM EAST (math convention), NOT clockwise from north.
        psi = 0       -> heading east
        psi = pi/2    -> heading north
        psi = pi      -> heading west
        psi = 3*pi/2  -> heading south
    Table 3 entries like "psi_0 = 0 deg" therefore correspond to an
    initial heading of due east, not due north.
"""
import numpy as np
import pytest

from src.common.aircraft import (
    AircraftParams,
    drag_force,
    fuel_flow,
    lift_coefficient,
    thrust_max,
)
from src.common.coordinates import LLAOrigin
from src.common.dynamics import dynamics_np, fuel_flow_np
from src.common.io_config import load_aircraft, load_wind_field
from src.common.wind_field import WindField, wind


@pytest.fixture(scope="module")
def ac() -> AircraftParams:
    return load_aircraft()


@pytest.fixture(scope="module")
def wf() -> WindField:
    return load_wind_field()


@pytest.fixture(scope="module")
def origin() -> LLAOrigin:
    return LLAOrigin(lambda0_deg=-4.0, phi0_deg=40.0)


@pytest.fixture(scope="module")
def zero_wind() -> WindField:
    """Wind field that returns 0 everywhere (clean dynamics tests)."""
    z = (0.0,) * 9
    return WindField(Wx_coeffs=z, Wy_coeffs=z)


def test_thrust_max_positive_at_cruise(ac: AircraftParams) -> None:
    T = thrust_max(8_000.0, ac)
    assert T > 0.0
    assert T < 200_000.0  # B737-800 has ~117 kN/engine


def test_thrust_max_decreases_with_altitude(ac: AircraftParams) -> None:
    h = np.linspace(0.0, 12_000.0, 13)
    T = np.array([thrust_max(hi, ac) for hi in h])
    assert np.all(np.diff(T) < 0.0)


def test_fuel_flow_nonnegative(ac: AircraftParams) -> None:
    f = fuel_flow(220.0, 9_000.0, 0.5, ac)
    assert f > 0.0


def test_fuel_flow_zero_at_zero_throttle(ac: AircraftParams) -> None:
    f = fuel_flow(220.0, 9_000.0, 0.0, ac)
    assert f == pytest.approx(0.0, abs=1e-12)


def test_lift_coefficient_in_cruise_band(ac: AircraftParams) -> None:
    # In level flight (mu=0), L = m*g => CL = 2mg/(rho S v^2).
    CL = lift_coefficient(70_000.0, 220.0, 9_000.0, 0.0, ac)
    assert 0.0 < CL < 1.0  # cruise CL is typically 0.3-0.6


def test_drag_force_positive(ac: AircraftParams) -> None:
    D = drag_force(70_000.0, 220.0, 9_000.0, 0.0, ac)
    assert D > 0.0


def test_dynamics_shape(ac: AircraftParams, wf: WindField, origin: LLAOrigin) -> None:
    state = np.array([0.0, 0.0, 8_000.0, 200.0, 0.0, 71_000.0])
    control = np.array([0.0, 0.0, 0.5])
    deriv = dynamics_np(state, control, ac, wf, origin)
    assert deriv.shape == (6,)


def test_dm_dt_strictly_negative_under_throttle(
    ac: AircraftParams, wf: WindField, origin: LLAOrigin
) -> None:
    state = np.array([0.0, 0.0, 8_000.0, 200.0, 0.0, 71_000.0])
    control = np.array([0.0, 0.0, 0.5])
    deriv = dynamics_np(state, control, ac, wf, origin)
    assert deriv[5] < 0.0, "Mass derivative must be negative when burning fuel"


def test_dm_dt_zero_at_zero_throttle(
    ac: AircraftParams, wf: WindField, origin: LLAOrigin
) -> None:
    state = np.array([0.0, 0.0, 8_000.0, 200.0, 0.0, 71_000.0])
    control = np.array([0.0, 0.0, 0.0])
    deriv = dynamics_np(state, control, ac, wf, origin)
    assert deriv[5] == pytest.approx(0.0, abs=1e-12)


def test_psi_zero_drives_eastward_no_wind(
    ac: AircraftParams, zero_wind: WindField, origin: LLAOrigin
) -> None:
    """psi=0, gamma=0, no wind -> dx=v, dy=0, dh=0."""
    v = 220.0
    state = np.array([0.0, 0.0, 9_000.0, v, 0.0, 70_000.0])
    control = np.array([0.0, 0.0, 0.5])
    deriv = dynamics_np(state, control, ac, zero_wind, origin)
    assert deriv[0] == pytest.approx(v, rel=1e-9)
    assert abs(deriv[1]) < 1e-9
    assert abs(deriv[2]) < 1e-9


def test_psi_pi_over_2_drives_northward_no_wind(
    ac: AircraftParams, zero_wind: WindField, origin: LLAOrigin
) -> None:
    """psi=pi/2, gamma=0, no wind -> dx=0, dy=v, dh=0."""
    v = 220.0
    state = np.array([0.0, 0.0, 9_000.0, v, np.pi / 2.0, 70_000.0])
    control = np.array([0.0, 0.0, 0.5])
    deriv = dynamics_np(state, control, ac, zero_wind, origin)
    assert abs(deriv[0]) < 1e-9
    assert deriv[1] == pytest.approx(v, rel=1e-9)
    assert abs(deriv[2]) < 1e-9


def test_psi_pi_over_2_under_wind_dx_equals_wx(
    ac: AircraftParams, wf: WindField, origin: LLAOrigin
) -> None:
    """psi=pi/2 (north) -> v contributes 0 to dx, so dx == Wx at origin."""
    state = np.array([0.0, 0.0, 9_000.0, 220.0, np.pi / 2.0, 70_000.0])
    control = np.array([0.0, 0.0, 0.5])
    deriv = dynamics_np(state, control, ac, wf, origin)
    Wx, _ = wind(origin.lambda0_deg, origin.phi0_deg, wf)
    assert deriv[0] == pytest.approx(Wx, rel=1e-9)


def test_climb_at_positive_gamma_no_wind(
    ac: AircraftParams, zero_wind: WindField, origin: LLAOrigin
) -> None:
    """gamma>0 -> dh>0; dh = v*sin(gamma)."""
    v = 220.0
    gamma = np.radians(3.0)
    state = np.array([0.0, 0.0, 9_000.0, v, 0.0, 70_000.0])
    control = np.array([gamma, 0.0, 0.5])
    deriv = dynamics_np(state, control, ac, zero_wind, origin)
    assert deriv[2] == pytest.approx(v * np.sin(gamma), rel=1e-9)


def test_fuel_flow_helper_matches_aircraft_module(ac: AircraftParams) -> None:
    state = np.array([0.0, 0.0, 9_000.0, 220.0, 0.0, 70_000.0])
    control = np.array([0.0, 0.0, 0.7])
    f1 = fuel_flow_np(state, control, ac)
    f2 = fuel_flow(220.0, 9_000.0, 0.7, ac)
    assert f1 == pytest.approx(f2, rel=1e-12)
