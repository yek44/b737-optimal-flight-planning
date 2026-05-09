"""Direct collocation NLP via compressed Hermite-Simpson + free `tf`.

State `x = (x_e, y_n, h, v, psi, m)`, control `u = (gamma, mu, delta)`,
free final time `tf`. Decision variables are *scaled* by `Scales` so
IPOPT sees O(1) magnitudes.

Cost (PDF eq. 7):
    J = (x(tf)-xd)^2 + (y(tf)-yd)^2 + (h(tf)-hd)^2
        + integral_0^tf (0.05 + f) dt

Compressed Hermite-Simpson (per segment k of length dt = tf/N):
    f_k    = f(x_k,   u_k)
    f_kp1  = f(x_kp1, u_kp1)
    u_kc   = (u_k + u_kp1) / 2                    [compressed: not a DV]
    x_kc   = 0.5*(x_k + x_kp1) + (dt/8)*(f_k - f_kp1)
    f_kc   = f(x_kc, u_kc)
    defect = x_kp1 - x_k - (dt/6)*(f_k + 4*f_kc + f_kp1)   == 0
    L_seg  = (dt/6) * (g_k + 4*g_kc + g_kp1)             (Simpson 1/3 rule)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import casadi as ca
import numpy as np

from ..common.aircraft import AircraftParams
from ..common.constants import DEG_TO_RAD
from ..common.cost import lagrange_integrand, mayer_terminal
from ..common.dynamics import dynamics_rhs
from ..common.io_config import FlightPlan
from ..common.results import FlightResult
from ..common.wind_field import WindField
from .initial_guess import build_initial_guess
from .scaling import SCALES


@dataclass
class CollocationResult:
    """Solver output before being wrapped into a `FlightResult`."""
    t: np.ndarray            # (N,)
    state: np.ndarray        # (N, 6)
    control: np.ndarray      # (N, 3)
    tf: float
    cost: float
    converged: bool
    iter_count: int
    return_status: str
    sol_x: np.ndarray        # raw scaled solution (for warm start)


def _ipopt_options(max_iter: int) -> dict:
    """Project-standard IPOPT options (CLAUDE.md §Optimization guardrails)."""
    return {
        "ipopt": {
            "linear_solver": "mumps",
            "tol": 1.0e-6,
            "mu_strategy": "adaptive",
            "hessian_approximation": "limited-memory",
            "max_iter": max_iter,
            "print_level": 0,
            "sb": "yes",
        },
        "print_time": False,
    }


def solve_direct(
    fp: FlightPlan,
    ac: AircraftParams,
    wf: WindField,
    n_nodes: int = 61,
    v_cruise_mps: float = 235.0,
    max_iter: int = 3000,
    warm_start: Optional[CollocationResult] = None,
) -> CollocationResult:
    """Build and solve the Hermite-Simpson NLP for one flight.

    Args:
        fp: Flight plan with initial conditions and destination.
        ac: B737-800 aircraft parameters.
        wf: Polynomial wind field.
        n_nodes: Number of collocation nodes (segments = n_nodes - 1).
        v_cruise_mps: Cruise airspeed used by the great-circle guess.
        max_iter: IPOPT iteration cap.
        warm_start: If provided, use its scaled solution as the initial
            guess (overrides the great-circle baseline). The grids must
            match (`n_nodes` and shapes).

    Returns:
        `CollocationResult` with the unscaled trajectory and metadata.
    """
    if n_nodes < 3 or (n_nodes - 1) % 2 != 0:
        # Hermite-Simpson defects use one segment at a time, so even
        # number of segments isn't strictly required, but it keeps the
        # node count sensible.
        pass

    guess = build_initial_guess(fp, ac, n_nodes=n_nodes, v_cruise_mps=v_cruise_mps)
    origin = guess.origin
    xd, yd, hd = guess.dest_enu

    sx = SCALES.state_array()
    su = SCALES.control_array()
    st_t = SCALES.t_s

    # ---- Decision variables (scaled) ----
    # Layout: [X_s (n_nodes*6), U_s (n_nodes*3), tf_s]
    X_s = ca.MX.sym("X_s", n_nodes, 6)
    U_s = ca.MX.sym("U_s", n_nodes, 3)
    tf_s = ca.MX.sym("tf_s")

    # Unscaled views (CasADi expressions).
    X = X_s * ca.repmat(ca.DM(sx).T, n_nodes, 1)
    U = U_s * ca.repmat(ca.DM(su).T, n_nodes, 1)
    tf = tf_s * st_t
    dt = tf / (n_nodes - 1)

    # ---- Helper closures (unscaled in, unscaled out) ----
    def f_dyn(xk, uk):
        return ca.vertcat(*dynamics_rhs(
            [xk[i] for i in range(6)],
            [uk[j] for j in range(3)],
            ac, wf, origin, math=ca,
        ))

    def g_run(xk, uk):
        return lagrange_integrand(xk[3], xk[2], uk[2], ac, math=ca)

    # ---- Defect constraints (compressed Hermite-Simpson) ----
    g_eq = []   # all collocation defects + initial-state pin
    J_run = ca.MX(0.0)
    for k in range(n_nodes - 1):
        xk, xkp1 = X[k, :].T, X[k + 1, :].T
        uk, ukp1 = U[k, :].T, U[k + 1, :].T

        fk = f_dyn(xk, uk)
        fkp1 = f_dyn(xkp1, ukp1)

        ukc = 0.5 * (uk + ukp1)
        xkc = 0.5 * (xk + xkp1) + (dt / 8.0) * (fk - fkp1)
        fkc = f_dyn(xkc, ukc)

        defect = xkp1 - xk - (dt / 6.0) * (fk + 4.0 * fkc + fkp1)
        # Scale the defect by 1/sx so each component is O(1) for IPOPT.
        g_eq.append(defect / ca.DM(sx))

        gk = g_run(xk, uk)
        gkc = g_run(xkc, ukc)
        gkp1 = g_run(xkp1, ukp1)
        J_run = J_run + (dt / 6.0) * (gk + 4.0 * gkc + gkp1)

    # ---- Initial-state pin (equality) ----
    psi0 = fp.initial.psi_deg * DEG_TO_RAD
    x0_phys = ca.DM([
        0.0, 0.0, fp.initial.h_m, fp.initial.v_mps, psi0, fp.initial.m_kg,
    ])
    g_eq.append((X[0, :].T - x0_phys) / ca.DM(sx))

    # ---- Terminal-state pin (hard equality on x, y, h) ----
    # We want the aircraft to actually arrive at (xd, yd, hd). The PDF's
    # squared-distance Mayer term is numerically nasty (huge gradient
    # magnitudes when scaled), so we enforce position+altitude as hard
    # equalities and report the squared terminal error in `cost` for
    # consistency with the PDF objective.
    xT = X[-1, 0]
    yT = X[-1, 1]
    hT = X[-1, 2]
    g_eq.append(ca.vertcat(
        (xT - xd) / sx[0],
        (yT - yd) / sx[1],
        (hT - hd) / sx[2],
    ))
    # `mayer_terminal(...)` would equal zero under the hard equality;
    # we recompute the PDF-style cost numerically after the solve.
    _ = mayer_terminal  # kept imported for documentation/symmetry

    # Objective sent to IPOPT is the running cost only; terminal error
    # is exactly zero by construction.
    J = J_run

    # ---- Bounds ----
    # State bounds (scaled).
    xs_lo = -np.inf * np.ones((n_nodes, 6))
    xs_hi = +np.inf * np.ones((n_nodes, 6))
    xs_lo[:, 2] = ac.h_min_m / sx[2]
    xs_hi[:, 2] = ac.h_max_m / sx[2]
    xs_lo[:, 3] = ac.v_min_mps / sx[3]
    xs_hi[:, 3] = ac.v_max_mps / sx[3]
    xs_lo[:, 5] = ac.m_OEW_kg / sx[5]
    xs_hi[:, 5] = fp.initial.m_kg / sx[5]

    us_lo = np.zeros((n_nodes, 3))
    us_hi = np.zeros((n_nodes, 3))
    us_lo[:, 0] = ac.gamma_min_deg * DEG_TO_RAD / su[0]
    us_hi[:, 0] = ac.gamma_max_deg * DEG_TO_RAD / su[0]
    us_lo[:, 1] = ac.mu_min_deg * DEG_TO_RAD / su[1]
    us_hi[:, 1] = ac.mu_max_deg * DEG_TO_RAD / su[1]
    us_lo[:, 2] = ac.delta_min / su[2]
    us_hi[:, 2] = ac.delta_max / su[2]

    tf_lo = 600.0 / st_t
    tf_hi = 18_000.0 / st_t

    # Pack decision vector.  CasADi `reshape(M, -1, 1)` is COLUMN-MAJOR
    # (Fortran order); numpy must match by passing `order='F'` below.
    z = ca.vertcat(ca.reshape(X_s, -1, 1), ca.reshape(U_s, -1, 1), tf_s)
    g = ca.vertcat(*g_eq)

    nlp = {"x": z, "f": J, "g": g}
    solver = ca.nlpsol("colloc", "ipopt", nlp, _ipopt_options(max_iter))

    # ---- Initial guess (scaled) ----
    if warm_start is not None and warm_start.sol_x.size == z.shape[0]:
        z0 = warm_start.sol_x.copy()
    else:
        Xs0 = guess.state / sx[None, :]
        Us0 = guess.control / su[None, :]
        z0 = np.concatenate([
            Xs0.reshape(-1, order="F"),
            Us0.reshape(-1, order="F"),
            [guess.tf / st_t],
        ])

    lbz = np.concatenate([
        xs_lo.reshape(-1, order="F"),
        us_lo.reshape(-1, order="F"),
        [tf_lo],
    ])
    ubz = np.concatenate([
        xs_hi.reshape(-1, order="F"),
        us_hi.reshape(-1, order="F"),
        [tf_hi],
    ])
    n_g = int(g.shape[0])

    sol = solver(
        x0=z0, lbx=lbz, ubx=ubz,
        lbg=np.zeros(n_g), ubg=np.zeros(n_g),
    )
    stats = solver.stats()

    z_opt = np.array(sol["x"]).flatten()
    Xs_opt = z_opt[: n_nodes * 6].reshape(n_nodes, 6, order="F")
    Us_opt = z_opt[n_nodes * 6: n_nodes * 6 + n_nodes * 3].reshape(
        n_nodes, 3, order="F"
    )
    tf_opt = float(z_opt[-1] * st_t)

    state_opt = Xs_opt * sx[None, :]
    control_opt = Us_opt * su[None, :]
    t_opt = np.linspace(0.0, tf_opt, n_nodes)

    # PDF objective (eq. 7) = Mayer terminal squared error + integral.
    # Terminal error is ~ 0 by hard constraint, so cost ~ J_run.
    pdf_cost = float(sol["f"]) + (
        (state_opt[-1, 0] - xd) ** 2
        + (state_opt[-1, 1] - yd) ** 2
        + (state_opt[-1, 2] - hd) ** 2
    )

    return CollocationResult(
        t=t_opt,
        state=state_opt,
        control=control_opt,
        tf=tf_opt,
        cost=pdf_cost,
        converged=bool(stats.get("success", False)),
        iter_count=int(stats.get("iter_count", -1)),
        return_status=str(stats.get("return_status", "unknown")),
        sol_x=z_opt,
    )


def to_flight_result(cr: CollocationResult, fp: FlightPlan, ac: AircraftParams,
                     n_nodes: int) -> FlightResult:
    """Wrap a `CollocationResult` into the canonical `FlightResult`."""
    fuel_total = float(fp.initial.m_kg - cr.state[-1, 5])
    return FlightResult(
        method="direct",
        flight_id=fp.id,
        t=cr.t,
        state=cr.state,
        control=cr.control,
        fuel_total_kg=fuel_total,
        duration_s=cr.tf,
        cost=cr.cost,
        converged=cr.converged,
        origin_lambda_deg=fp.initial.lambda_deg,
        origin_phi_deg=fp.initial.phi_deg,
        solver_meta={
            "scheme": "hermite-simpson-compressed",
            "n_nodes": int(n_nodes),
            "iter_count": cr.iter_count,
            "return_status": cr.return_status,
            "ipopt_linear_solver": "mumps",
        },
    )
