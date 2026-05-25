"""The 8 standard plots required by the project, plus method-overlay helpers.

The PDF lists:
    x-y, h-Time, V_tas-Time, Mass-Time, Thrust-Time, Throttle-Time,
    Bank angle-Time, Flight path angle-Time.

Plot policy (see CLAUDE.md): every axis labelled with units in [brackets],
legends on multi-line plots, no bare titles.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np

from .aircraft import AircraftParams, thrust_max
from .constants import RAD_TO_DEG
from .results import FlightResult


# Plot file naming convention -- DO NOT change without updating compare/.
PLOT_NAMES = (
    "xy",
    "h_time",
    "v_time",
    "mass_time",
    "thrust_time",
    "throttle_time",
    "mu_time",
    "gamma_time",
)


def _figure(figsize=(8, 4.5)) -> tuple[plt.Figure, plt.Axes]:
    """Build a figure with a sensible default layout.

    The default ``(8, 4.5)`` aspect is tuned for time-series plots that
    sit at ``0.48\\linewidth`` in the report --- it keeps tick labels
    legible and gives axis text enough room to breathe. For x-y ground
    tracks use :func:`_xy_figsize` to pick a track-aware aspect.
    """
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.grid(True, alpha=0.3)
    return fig, ax


def _xy_figsize(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pick a track-aware ``(w, h)`` figsize for an x-y ground track.

    Long-thin trajectories (e.g. Flight 3, which is almost purely
    northbound) get a tall narrow canvas so an ``equal``-aspect plot
    does not waste space; near-square trajectories get a square canvas.
    The bound on each side keeps the rendered PNG at a sensible size
    for inclusion in the report.
    """
    dx = float(np.ptp(x))
    dy = float(np.ptp(y))
    if dx <= 0.0:
        dx = 1.0
    ratio = dy / dx
    if ratio > 2.0:
        return (4.5, 8.0)
    if ratio > 1.3:
        return (5.5, 7.5)
    if ratio < 0.5:
        return (10.0, 4.5)
    if ratio < 0.77:
        return (9.0, 5.5)
    return (7.0, 6.0)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _control_aligned_to_time(t: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align control history to the time vector for plotting.

    If `u` has shape `(N-1, 3)` (per-segment), repeat each row at its
    segment start; if `(N, 3)`, return as-is.
    """
    if u.shape[0] == t.shape[0]:
        return t, u
    if u.shape[0] == t.shape[0] - 1:
        # Step plot: control held constant over each segment.
        return t, np.vstack([u, u[-1:]])
    raise ValueError(f"control shape {u.shape} incompatible with time shape {t.shape}")


def plot_xy(result: FlightResult, ax: Optional[plt.Axes] = None,
            label: Optional[str] = None) -> plt.Axes:
    """x-y trajectory in ENU meters, with destination marker."""
    own = ax is None
    x = result.state[:, 0]
    y = result.state[:, 1]
    if own:
        _, ax = _figure(figsize=_xy_figsize(x, y))
    assert ax is not None
    ax.plot(x, y, label=label or result.method)
    ax.scatter([x[0]], [y[0]], marker="o", color="black", zorder=5, label="start" if own else None)
    ax.scatter([x[-1]], [y[-1]], marker="x", color="red", zorder=5, label="end" if own else None)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Flight {result.flight_id} -- x-y trajectory")
    if own:
        ax.legend()
    return ax


def plot_h_time(result: FlightResult, ax: Optional[plt.Axes] = None,
                label: Optional[str] = None) -> plt.Axes:
    own = ax is None
    if own:
        _, ax = _figure()
    assert ax is not None
    ax.plot(result.t, result.state[:, 2], label=label or result.method)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(f"Flight {result.flight_id} -- altitude vs time")
    if own:
        ax.legend()
    return ax


def plot_v_time(result: FlightResult, ax: Optional[plt.Axes] = None,
                label: Optional[str] = None) -> plt.Axes:
    own = ax is None
    if own:
        _, ax = _figure()
    assert ax is not None
    ax.plot(result.t, result.state[:, 3], label=label or result.method)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("TAS [m/s]")
    ax.set_title(f"Flight {result.flight_id} -- airspeed vs time")
    if own:
        ax.legend()
    return ax


def plot_mass_time(result: FlightResult, ax: Optional[plt.Axes] = None,
                   label: Optional[str] = None) -> plt.Axes:
    own = ax is None
    if own:
        _, ax = _figure()
    assert ax is not None
    ax.plot(result.t, result.state[:, 5], label=label or result.method)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass [kg]")
    ax.set_title(f"Flight {result.flight_id} -- mass vs time")
    if own:
        ax.legend()
    return ax


def plot_thrust_time(result: FlightResult, ac: AircraftParams,
                     ax: Optional[plt.Axes] = None,
                     label: Optional[str] = None) -> plt.Axes:
    own = ax is None
    if own:
        _, ax = _figure()
    assert ax is not None
    t, u = _control_aligned_to_time(result.t, result.control)
    h = result.state[:, 2]
    delta = u[:, 2]
    Tmax = thrust_max(h, ac, math=np)
    T = Tmax * delta
    ax.plot(t, T, label=label or result.method)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title(f"Flight {result.flight_id} -- thrust vs time")
    if own:
        ax.legend()
    return ax


def plot_throttle_time(result: FlightResult, ax: Optional[plt.Axes] = None,
                       label: Optional[str] = None) -> plt.Axes:
    own = ax is None
    if own:
        _, ax = _figure()
    assert ax is not None
    t, u = _control_aligned_to_time(result.t, result.control)
    ax.plot(t, u[:, 2], label=label or result.method)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Throttle [-]")
    ax.set_title(f"Flight {result.flight_id} -- throttle vs time")
    if own:
        ax.legend()
    return ax


def plot_mu_time(result: FlightResult, ax: Optional[plt.Axes] = None,
                 label: Optional[str] = None) -> plt.Axes:
    own = ax is None
    if own:
        _, ax = _figure()
    assert ax is not None
    t, u = _control_aligned_to_time(result.t, result.control)
    ax.plot(t, u[:, 1] * RAD_TO_DEG, label=label or result.method)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bank angle [deg]")
    ax.set_title(f"Flight {result.flight_id} -- bank angle vs time")
    if own:
        ax.legend()
    return ax


def plot_gamma_time(result: FlightResult, ax: Optional[plt.Axes] = None,
                    label: Optional[str] = None) -> plt.Axes:
    own = ax is None
    if own:
        _, ax = _figure()
    assert ax is not None
    t, u = _control_aligned_to_time(result.t, result.control)
    ax.plot(t, u[:, 0] * RAD_TO_DEG, label=label or result.method)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flight path angle [deg]")
    ax.set_title(f"Flight {result.flight_id} -- flight path angle vs time")
    if own:
        ax.legend()
    return ax


def make_eight_plots(result: FlightResult, ac: AircraftParams,
                     outdir: Path | str) -> List[Path]:
    """Render the 8 required plots for a single flight to `outdir`.

    Returns the list of saved paths in `PLOT_NAMES` order.
    """
    out = Path(outdir)
    paths: List[Path] = []
    plotters = (
        ("xy", lambda: plot_xy(result)),
        ("h_time", lambda: plot_h_time(result)),
        ("v_time", lambda: plot_v_time(result)),
        ("mass_time", lambda: plot_mass_time(result)),
        ("thrust_time", lambda: plot_thrust_time(result, ac)),
        ("throttle_time", lambda: plot_throttle_time(result)),
        ("mu_time", lambda: plot_mu_time(result)),
        ("gamma_time", lambda: plot_gamma_time(result)),
    )
    for name, fn in plotters:
        ax = fn()
        path = out / f"{name}.png"
        fig = ax.get_figure()
        assert fig is not None
        _save(cast(plt.Figure, fig), path)
        paths.append(path)
    return paths


def overlay_methods(
    results: Sequence[FlightResult],
    ac: AircraftParams,
    outdir: Path | str,
) -> List[Path]:
    """Render comparison overlays (one per metric) across multiple methods.

    All `results` must be for the same flight_id. One legend per figure.
    """
    if not results:
        raise ValueError("overlay_methods needs >=1 results")
    fid = results[0].flight_id
    if any(r.flight_id != fid for r in results):
        raise ValueError("overlay_methods got mixed flight ids")

    out = Path(outdir)
    paths: List[Path] = []

    def _do(name: str, plot_fn, figsize=None) -> None:
        fig, ax = _figure() if figsize is None else _figure(figsize=figsize)
        for r in results:
            plot_fn(r, ax=ax, label=r.method)
        ax.legend()
        path = out / f"{name}.png"
        _save(fig, path)
        paths.append(path)

    # x-y overlay uses a track-aware figsize derived from the first
    # result so long-thin trajectories (e.g. Flight 3) get a tall canvas.
    _xy = _xy_figsize(results[0].state[:, 0], results[0].state[:, 1])
    _do("xy", plot_xy, figsize=_xy)
    _do("h_time", plot_h_time)
    _do("v_time", plot_v_time)
    _do("mass_time", plot_mass_time)
    _do("throttle_time", plot_throttle_time)
    _do("mu_time", plot_mu_time)
    _do("gamma_time", plot_gamma_time)

    # thrust overlay needs ac
    fig, ax = _figure()
    for r in results:
        plot_thrust_time(r, ac, ax=ax, label=r.method)
    ax.legend()
    p = out / "thrust_time.png"
    _save(fig, p)
    paths.append(p)

    return paths
