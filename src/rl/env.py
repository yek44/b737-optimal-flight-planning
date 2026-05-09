"""Gymnasium environment wrapping the 6-state aircraft dynamics.

State-space env with continuous action `(gamma, mu, delta)` rescaled
from the policy's `[-1, 1]^3` output. Observation is a normalized
10-vector that includes ENU error to the destination, so the policy
can generalize across flight scenarios.

Reward:
    r = -(0.05 + f) * dt - 1e-3 * dist_to_dest
        + terminal_bonus_if_reached - violation_penalty_if_oob
The first term mirrors the PDF Lagrangian; the small distance penalty
provides shaping; bounds violations terminate with a stiff penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..common.aircraft import AircraftParams
from ..common.constants import DEG_TO_RAD
from ..common.coordinates import LLAOrigin, lla_to_enu
from ..common.dynamics import dynamics_np
from ..common.io_config import (
    FlightPlan,
    get_flight_plan,
    load_aircraft,
    load_wind_field,
)
from ..common.wind_field import WindField


def _rk4_step(state: np.ndarray, control: np.ndarray, dt: float,
              ac: AircraftParams, wf: WindField, origin: LLAOrigin) -> np.ndarray:
    """Single RK4 step of the 6-state ODE."""
    k1 = dynamics_np(state, control, ac, wf, origin)
    k2 = dynamics_np(state + 0.5 * dt * k1, control, ac, wf, origin)
    k3 = dynamics_np(state + 0.5 * dt * k2, control, ac, wf, origin)
    k4 = dynamics_np(state + dt * k3, control, ac, wf, origin)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@dataclass
class EnvConfig:
    """Tunable env parameters."""
    dt_s: float = 10.0
    max_steps: int = 2400        # ~ 4 h horizon at dt=10
    terminal_radius_m: float = 8_000.0
    terminal_h_tol_m: float = 1000.0
    distance_shaping: float = 1.0e-2     # ~ 20 reward per fast step
    fuel_weight: float = 0.2             # de-emphasised vs. shaping
    time_weight: float = 0.01
    violation_penalty: float = 50.0      # softened from 100
    terminal_bonus: float = 500.0


class FlightPlannerEnv(gym.Env):
    """Custom env for cruise-phase trajectory optimisation.

    Action: 3D continuous in [-1, 1] -> (gamma, mu, delta).
    Observation: 10D normalised vector
        [x_err/sx, y_err/sx, h_err/sh, h/sh, v/sv, sin(psi), cos(psi),
         m_norm, t_frac, dist_norm].
    """
    metadata = {"render_modes": []}

    def __init__(self, flight_id: int = 1,
                 cfg: Optional[EnvConfig] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.flight_id = int(flight_id)
        self.fp: FlightPlan = get_flight_plan(self.flight_id)
        self.ac: AircraftParams = load_aircraft()
        self.wf: WindField = load_wind_field()
        self.origin = LLAOrigin(self.fp.initial.lambda_deg,
                                self.fp.initial.phi_deg)
        xd, yd = lla_to_enu(self.fp.destination.lambda_deg,
                            self.fp.destination.phi_deg, self.origin)
        self.dest_xyz = np.array([float(xd), float(yd),
                                  self.fp.destination.h_m])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(10,), dtype=np.float32)

        # Action affine map: action in [-1,1]^3 -> physical bounds.
        self._u_lo = np.array([
            self.ac.gamma_min_deg * DEG_TO_RAD,
            self.ac.mu_min_deg * DEG_TO_RAD,
            self.ac.delta_min,
        ])
        self._u_hi = np.array([
            self.ac.gamma_max_deg * DEG_TO_RAD,
            self.ac.mu_max_deg * DEG_TO_RAD,
            self.ac.delta_max,
        ])

        self._sx = np.array([1.0e6, 1.0e6, 1.0e4])  # x, y, h normalisation
        self._sv = 250.0
        self._sh = 1.0e4

        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        self.state = np.zeros(6)
        self._init_dist = 1.0
        self.history_state: list[np.ndarray] = []
        self.history_control: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        psi0 = self.fp.initial.psi_deg * DEG_TO_RAD
        self.state = np.array([
            0.0, 0.0,
            self.fp.initial.h_m,
            self.fp.initial.v_mps,
            psi0,
            self.fp.initial.m_kg,
        ], dtype=float)
        self._step_idx = 0
        self._init_dist = float(np.linalg.norm(self.state[:2] - self.dest_xyz[:2]))
        self.history_state = [self.state.copy()]
        self.history_control = []
        return self._observe(), {}

    def _to_physical(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a, -1.0, 1.0)
        return self._u_lo + 0.5 * (a + 1.0) * (self._u_hi - self._u_lo)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        u = self._to_physical(np.asarray(action, dtype=float))
        self.history_control.append(u.copy())
        s_prev = self.state.copy()
        self.state = _rk4_step(self.state, u, self.cfg.dt_s,
                               self.ac, self.wf, self.origin)
        self.history_state.append(self.state.copy())
        self._step_idx += 1

        # Per-step PDF-style cost.
        from ..common.aircraft import fuel_flow as _ff
        f = float(_ff(s_prev[3], s_prev[2], u[2], self.ac))
        running = (self.cfg.time_weight + self.cfg.fuel_weight * f) * self.cfg.dt_s
        reward = -running

        # Distance shaping: dense penalty proportional to current
        # distance + closing-rate bonus. The dense penalty stops the
        # agent from overshooting (no benefit to moving past the
        # destination); the closing bonus accelerates early learning.
        d_prev = float(np.linalg.norm(s_prev[:2] - self.dest_xyz[:2]))
        d_now = float(np.linalg.norm(self.state[:2] - self.dest_xyz[:2]))
        reward += self.cfg.distance_shaping * (d_prev - d_now)
        reward -= 1.0e-5 * d_now * self.cfg.dt_s

        terminated = False
        truncated = False
        violation = False
        x, y, h, v, psi, m = self.state
        # Soft bounds: clip altitude and airspeed back to envelope and
        # apply a per-step penalty proportional to the excursion. This
        # gives PPO useful gradient on bounds rather than killing the
        # episode on the first slip.
        if h < self.ac.h_min_m:
            reward -= self.cfg.violation_penalty * 0.1
            h = self.ac.h_min_m
            self.state[2] = h
            violation = True
        elif h > self.ac.h_max_m:
            reward -= self.cfg.violation_penalty * 0.1
            h = self.ac.h_max_m
            self.state[2] = h
            violation = True
        if v < self.ac.v_min_mps:
            reward -= self.cfg.violation_penalty * 0.1
            v = self.ac.v_min_mps
            self.state[3] = v
            violation = True
        elif v > self.ac.v_max_mps:
            reward -= self.cfg.violation_penalty * 0.1
            v = self.ac.v_max_mps
            self.state[3] = v
            violation = True
        if m < self.ac.m_OEW_kg:
            terminated = True
            reward -= self.cfg.violation_penalty

        if (not violation and d_now <= self.cfg.terminal_radius_m
                and abs(h - self.dest_xyz[2]) <= self.cfg.terminal_h_tol_m):
            reward += self.cfg.terminal_bonus
            terminated = True

        if self._step_idx >= self.cfg.max_steps:
            truncated = True

        info = {
            "dist_m": d_now,
            "h_err_m": h - self.dest_xyz[2],
            "violation": violation,
            "fuel_kg_step": f * self.cfg.dt_s,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        x, y, h, v, psi, m = self.state
        err = self.dest_xyz - np.array([x, y, h])
        m_frac = (m - self.ac.m_OEW_kg) / max(1.0, self.fp.initial.m_kg - self.ac.m_OEW_kg)
        t_frac = self._step_idx / max(1, self.cfg.max_steps)
        dist = float(np.linalg.norm(err[:2]))
        dist_norm = dist / max(1.0, self._init_dist)
        obs = np.array([
            err[0] / self._sx[0],
            err[1] / self._sx[1],
            err[2] / self._sh,
            h / self._sh,
            v / self._sv,
            np.sin(psi),
            np.cos(psi),
            float(np.clip(m_frac, 0.0, 1.0)),
            float(t_frac),
            float(dist_norm),
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    # Helpers used by `evaluate.py`
    # ------------------------------------------------------------------
    def trajectory_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (t, state, control) for the last completed episode."""
        state = np.asarray(self.history_state)
        control = np.asarray(self.history_control) if self.history_control else np.zeros((0, 3))
        n = state.shape[0]
        t = np.arange(n) * self.cfg.dt_s
        return t, state, control

    def make_factory(self, seed: int = 0):
        """Return a callable that creates a fresh env (for SubprocVecEnv)."""
        flight_id = self.flight_id
        cfg = self.cfg

        def _f() -> FlightPlannerEnv:
            return FlightPlannerEnv(flight_id=flight_id, cfg=cfg, seed=seed)
        return _f
