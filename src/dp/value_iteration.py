"""Numba-jitted value iteration on a 2-D position grid.

We solve the Bellman equation
    V(x, y) = min_psi [ c(x, y, psi) * dt + V(x', y') ]
where (x', y') is the kinematic step under (v_cruise, Wx, Wy, psi)
projected back into the grid via bilinear interpolation. Terminal
cells (within `r_term` of the destination) are absorbing with V = 0.

The wind field is precomputed on the same (x, y) grid using the
per-flight ENU origin; this turns Wx, Wy lookups into O(1) array
indexing inside the inner Bellman loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class ValueGrid:
    """Container for the DP solution on a 2-D ENU grid."""
    x_edges: np.ndarray   # (Nx+1,) cell edges [m]
    y_edges: np.ndarray   # (Ny+1,) cell edges [m]
    V: np.ndarray         # (Nx, Ny) value function
    Wx: np.ndarray        # (Nx, Ny) east wind [m/s] at cell centers
    Wy: np.ndarray        # (Nx, Ny) north wind [m/s] at cell centers
    headings: np.ndarray  # (n_psi,) candidate headings [rad]
    dest_xy: tuple[float, float]
    r_term_m: float
    dt_s: float
    v_mps: float
    cost_rate: float      # 0.05 + f_const, the per-second running cost


@njit(cache=True, fastmath=True)
def _bilinear(arr, x_edges, y_edges, x, y):
    """Bilinear interpolation of a 2-D array at point (x, y).

    Out-of-grid points return a large penalty value so the optimiser
    avoids them.
    """
    nx = arr.shape[0]
    ny = arr.shape[1]
    if x < x_edges[0] or x > x_edges[-1] or y < y_edges[0] or y > y_edges[-1]:
        return 1.0e18
    # Cell index: edges have nx+1 entries. cell_x in [0, nx-1].
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    fx = (x - x_edges[0]) / dx - 0.5
    fy = (y - y_edges[0]) / dy - 0.5
    i = int(np.floor(fx))
    j = int(np.floor(fy))
    if i < 0:
        i = 0
    if j < 0:
        j = 0
    if i > nx - 2:
        i = nx - 2
    if j > ny - 2:
        j = ny - 2
    a = fx - i
    b = fy - j
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    if b < 0.0:
        b = 0.0
    if b > 1.0:
        b = 1.0
    return (
        (1.0 - a) * (1.0 - b) * arr[i, j]
        + a * (1.0 - b) * arr[i + 1, j]
        + (1.0 - a) * b * arr[i, j + 1]
        + a * b * arr[i + 1, j + 1]
    )


@njit(cache=True, fastmath=True)
def _bellman_sweep(V, Wx, Wy, x_edges, y_edges, headings, dt, v,
                   cost_rate, dest_x, dest_y, r_term):
    """One full Gauss-Seidel sweep of the Bellman backup."""
    nx, ny = V.shape
    n_psi = headings.shape[0]
    max_delta = 0.0
    for i in range(nx):
        x = 0.5 * (x_edges[i] + x_edges[i + 1])
        for j in range(ny):
            y = 0.5 * (y_edges[j] + y_edges[j + 1])
            # Absorbing terminal disk.
            if (x - dest_x) ** 2 + (y - dest_y) ** 2 <= r_term * r_term:
                V[i, j] = 0.0
                continue
            best = 1.0e30
            wx = Wx[i, j]
            wy = Wy[i, j]
            step_cost = cost_rate * dt
            for k in range(n_psi):
                psi = headings[k]
                xn = x + (v * np.cos(psi) + wx) * dt
                yn = y + (v * np.sin(psi) + wy) * dt
                vn = _bilinear(V, x_edges, y_edges, xn, yn)
                val = step_cost + vn
                if val < best:
                    best = val
            old = V[i, j]
            V[i, j] = best
            d = old - best
            if d < 0.0:
                d = -d
            if d > max_delta:
                max_delta = d
    return max_delta


def precompute_wind_grid(x_edges, y_edges, origin, wf):
    """Evaluate Wx, Wy at every cell center on the ENU grid."""
    from ..common.coordinates import enu_to_lla
    from ..common.wind_field import wind
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    lam, phi = enu_to_lla(XC, YC, origin)
    Wx, Wy = wind(lam, phi, wf)
    return Wx.astype(np.float64), Wy.astype(np.float64)


def solve_value_iteration(
    *,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    Wx: np.ndarray,
    Wy: np.ndarray,
    headings: np.ndarray,
    dest_xy: tuple[float, float],
    r_term_m: float,
    dt_s: float,
    v_mps: float,
    cost_rate: float,
    max_sweeps: int = 800,
    tol: float = 1.0e-2,
    verbose: bool = False,
) -> ValueGrid:
    """Run Gauss-Seidel value iteration until `max_delta` < `tol`."""
    nx = x_edges.size - 1
    ny = y_edges.size - 1
    V = np.full((nx, ny), 1.0e10, dtype=np.float64)

    # Seed terminal disk.
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    dx_dest, dy_dest = dest_xy
    DX = xc[:, None] - dx_dest
    DY = yc[None, :] - dy_dest
    inside = DX * DX + DY * DY <= r_term_m * r_term_m
    V[inside] = 0.0

    for sweep in range(max_sweeps):
        delta = _bellman_sweep(V, Wx, Wy, x_edges, y_edges, headings,
                               dt_s, v_mps, cost_rate,
                               dest_xy[0], dest_xy[1], r_term_m)
        if verbose and (sweep % 25 == 0 or delta < tol):
            print(f"  [VI] sweep {sweep:4d}  max_delta={delta:.3e}")
        if delta < tol:
            break

    return ValueGrid(
        x_edges=x_edges, y_edges=y_edges, V=V, Wx=Wx, Wy=Wy,
        headings=headings, dest_xy=dest_xy, r_term_m=r_term_m,
        dt_s=dt_s, v_mps=v_mps, cost_rate=cost_rate,
    )


def rollout_greedy(vg: ValueGrid, start_xy: tuple[float, float],
                   max_steps: int = 5000) -> np.ndarray:
    """Greedy rollout: at each state pick the action minimising V'.

    Returns a (T, 2) array of (x, y) positions. Stops when the
    aircraft enters the terminal disk or `max_steps` is exceeded.
    """
    x, y = start_xy
    out = [(x, y)]
    nx, ny = vg.V.shape
    dx_cell = vg.x_edges[1] - vg.x_edges[0]
    dy_cell = vg.y_edges[1] - vg.y_edges[0]
    for _ in range(max_steps):
        if (x - vg.dest_xy[0]) ** 2 + (y - vg.dest_xy[1]) ** 2 <= vg.r_term_m ** 2:
            break
        # Wind at current cell (nearest center).
        i = int(np.clip((x - vg.x_edges[0]) / dx_cell - 0.5, 0, nx - 1))
        j = int(np.clip((y - vg.y_edges[0]) / dy_cell - 0.5, 0, ny - 1))
        wx = vg.Wx[i, j]
        wy = vg.Wy[i, j]
        best_val = np.inf
        best_psi = 0.0
        for psi in vg.headings:
            xn = x + (vg.v_mps * np.cos(psi) + wx) * vg.dt_s
            yn = y + (vg.v_mps * np.sin(psi) + wy) * vg.dt_s
            vn = _bilinear(vg.V, vg.x_edges, vg.y_edges, xn, yn)
            val = vg.cost_rate * vg.dt_s + vn
            if val < best_val:
                best_val = val
                best_psi = psi
        x = x + (vg.v_mps * np.cos(best_psi) + wx) * vg.dt_s
        y = y + (vg.v_mps * np.sin(best_psi) + wy) * vg.dt_s
        out.append((x, y))
    return np.asarray(out)
