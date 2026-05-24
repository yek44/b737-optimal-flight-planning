"""Gymnasium environment wrapping the 6-state aircraft dynamics.

State-space env with continuous action `(gamma, mu, delta)` rescaled
from the policy's `[-1, 1]^3` output. Observation is a normalised
21-vector that includes:

* 12 base channels: ENU error to the destination, current altitude,
  airspeed, heading (sin/cos), mass fraction, time fraction, distance
  fraction, and local wind (Wx, Wy)/50.
* 9 action-history channels: the last 3 actions, oldest first. Action
  history was the key change that enabled the policy to imitate the
  bang-bang bank manoeuvres of Flight 2 and to recover from RK4 drift
  on Flight 3 -- without it BC alone could not predict the next action
  from the instantaneous state in long-horizon scenarios.

Reward shaping is per-flight tunable via :class:`EnvConfig` (see
:func:`config_for_flight`):

* PDF integrand ``r_run = -(time_weight + f_kg_per_s) * dt``
* Closing-rate shaping ``+ closing_weight * (d_prev - d_now)`` capped
  inside the approach corridor so the policy is not rewarded for
  blasting through the destination disk.
* Altitude corridor regulariser around a linear ramp ``h_ref(t)``.
* Soft envelope clipping with bounded per-step penalty.
* Stiff terminal bonus on disk entry, shrunken bonus on sustained
  overshoot or time-out (so the policy retains gradient even if the
  episode ends short of the disk).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

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


# Action-history length baked into the observation.  Three is enough
# to recover instantaneous control rate and acceleration on the 10 s
# step grid without inflating the observation dimension.
ACTION_HISTORY_LEN = 3
OBS_DIM = 12 + 3 * ACTION_HISTORY_LEN  # 21


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
    """Tunable env parameters.

    Reward design (balances PDF cost with practical learnability):
        r = -(time_weight + fuel_weight * f) * dt    PDF-style running
          + closing_weight * (d_prev - d_now)        dense closing
          - altitude_weight * |h - h_ref(t)|         altitude regulariser
          - ground_penalty * max(0, h_floor - h)     ground-skim deterrent
          + terminal_bonus_if_in_disk                large sparse goal
          + terminal_bonus * shrink                  shaped overshoot bonus
          - violation_penalty per OOB step           soft envelope penalty

    The defaults are tuned for Flight 1 (long, smooth cruise).
    Per-flight overrides live in :func:`config_for_flight`.
    """
    dt_s: float = 10.0
    max_steps: int = 2400
    terminal_radius_m: float = 10_000.0
    terminal_h_tol_m: float = 1500.0
    fuel_weight: float = 1.0
    time_weight: float = 0.05
    closing_weight: float = 2.0e-3
    altitude_weight: float = 5.0e-4
    ground_penalty: float = 0.01
    ground_floor_m: float = 4000.0
    violation_penalty: float = 5.0
    terminal_bonus: float = 20000.0
    overshoot_corridor_factor: float = 10.0
    gamma: float = 0.995


def config_for_flight(flight_id: int) -> EnvConfig:
    """Per-flight overrides on top of the default ``EnvConfig``.

    Flight 2 (short, sharp-turn, high-bank): tighter closing reward and
    lower altitude regularisation -- the optimum here banks hard early
    rather than holding altitude. Flight 3 (long, 90-degree heading
    change): much higher closing weight + a longer max horizon so the
    policy is not truncated short of the destination.
    """
    cfg = EnvConfig()
    if flight_id == 2:
        cfg.closing_weight = 6.0e-3
        cfg.altitude_weight = 2.0e-4
        cfg.max_steps = 1800
    elif flight_id == 3:
        cfg.closing_weight = 4.0e-3
        cfg.altitude_weight = 3.0e-4
        cfg.max_steps = 3000
    return cfg


class FlightPlannerEnv(gym.Env):
    """Custom env for cruise-phase trajectory optimisation.

    Action: 3D continuous in [-1, 1] -> (gamma, mu, delta).
    Observation: 21D normalised vector (see module docstring).
    """
    metadata = {"render_modes": []}

    def __init__(self, flight_id: int = 1,
                 cfg: Optional[EnvConfig] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg or config_for_flight(flight_id)
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
                                            shape=(OBS_DIM,),
                                            dtype=np.float32)

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

        self._sx = np.array([1.0e6, 1.0e6, 1.0e4])
        self._sv = 250.0
        self._sh = 1.0e4

        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        self.state = np.zeros(6)
        self._init_dist = 1.0
        self._init_h = float(self.fp.initial.h_m)
        self._dest_h = float(self.fp.destination.h_m)
        self.history_state: list[np.ndarray] = []
        self.history_control: list[np.ndarray] = []
        self._action_history: Deque[np.ndarray] = deque(
            [np.zeros(3, dtype=np.float32)] * ACTION_HISTORY_LEN,
            maxlen=ACTION_HISTORY_LEN,
        )
        self._closest_dist = float("inf")
        self._prev_dist = 1.0

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset to the (possibly perturbed) initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        psi0 = self.fp.initial.psi_deg * DEG_TO_RAD
        x0 = np.array([
            0.0, 0.0,
            self.fp.initial.h_m,
            self.fp.initial.v_mps,
            psi0,
            self.fp.initial.m_kg,
        ], dtype=float)
        # Optional initial-state perturbation (used by DAgger rollouts).
        if options is not None and "initial_state" in options:
            x0 = np.asarray(options["initial_state"], dtype=float).copy()
        self.state = x0
        self._step_idx = 0
        self._init_dist = float(np.linalg.norm(self.state[:2] - self.dest_xyz[:2]))
        self.history_state = [self.state.copy()]
        self.history_control = []
        self._action_history = deque(
            [np.zeros(3, dtype=np.float32)] * ACTION_HISTORY_LEN,
            maxlen=ACTION_HISTORY_LEN,
        )
        self._closest_dist = self._init_dist
        return self._observe(), {}

    def _to_physical(self, a: np.ndarray) -> np.ndarray:
        a = np.clip(a, -1.0, 1.0)
        return self._u_lo + 0.5 * (a + 1.0) * (self._u_hi - self._u_lo)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one RK4 step with the given normalised action."""
        a_in = np.asarray(action, dtype=np.float32).reshape(3)
        u = self._to_physical(a_in.astype(float))
        self.history_control.append(u.copy())
        s_prev = self.state.copy()
        self.state = _rk4_step(self.state, u, self.cfg.dt_s,
                               self.ac, self.wf, self.origin)
        self.history_state.append(self.state.copy())
        self._step_idx += 1

        from ..common.aircraft import fuel_flow as _ff
        f = float(_ff(s_prev[3], s_prev[2], u[2], self.ac))
        running = (self.cfg.time_weight + self.cfg.fuel_weight * f) * self.cfg.dt_s
        reward = -running

        d_prev = float(np.linalg.norm(s_prev[:2] - self.dest_xyz[:2]))
        d_now = float(np.linalg.norm(self.state[:2] - self.dest_xyz[:2]))
        cap = self.cfg.overshoot_corridor_factor * self.cfg.terminal_radius_m
        if d_prev > cap or d_now > cap:
            reward += self.cfg.closing_weight * (d_prev - d_now)

        frac = self._step_idx / max(1, self.cfg.max_steps)
        h_ref = (1.0 - frac) * self._init_h + frac * self._dest_h
        reward -= self.cfg.altitude_weight * abs(self.state[2] - h_ref)
        if self.state[2] < self.cfg.ground_floor_m:
            reward -= self.cfg.ground_penalty * (
                self.cfg.ground_floor_m - self.state[2]
            )

        terminated = False
        truncated = False
        violation = False
        x, y, h, v, psi, m = self.state
        if h < self.ac.h_min_m:
            reward -= self.cfg.violation_penalty
            h = self.ac.h_min_m
            self.state[2] = h
            violation = True
        elif h > self.ac.h_max_m:
            reward -= self.cfg.violation_penalty
            h = self.ac.h_max_m
            self.state[2] = h
            violation = True
        if v < self.ac.v_min_mps:
            reward -= self.cfg.violation_penalty
            v = self.ac.v_min_mps
            self.state[3] = v
            violation = True
        elif v > self.ac.v_max_mps:
            reward -= self.cfg.violation_penalty
            v = self.ac.v_max_mps
            self.state[3] = v
            violation = True
        if m < self.ac.m_OEW_kg:
            terminated = True
            reward -= self.cfg.violation_penalty * 20.0

        if d_now < self._closest_dist:
            self._closest_dist = d_now
        approach_corridor = self.cfg.overshoot_corridor_factor * self.cfg.terminal_radius_m
        in_disk = (d_now <= self.cfg.terminal_radius_m
                   and abs(h - self.dest_xyz[2]) <= self.cfg.terminal_h_tol_m)
        sustained_overshoot = (
            self._closest_dist < approach_corridor
            and d_now > 1.5 * self._closest_dist
            and d_now > self._closest_dist + 5.0e3
        )
        if in_disk or sustained_overshoot:
            if in_disk:
                reward += self.cfg.terminal_bonus
            else:
                shrink = max(0.0, 1.0 - self._closest_dist / approach_corridor)
                reward += self.cfg.terminal_bonus * shrink
            terminated = True

        if self._step_idx >= self.cfg.max_steps:
            truncated = True
            if not in_disk and not sustained_overshoot:
                shrink = max(0.0, 1.0 - self._closest_dist / approach_corridor)
                reward += self.cfg.terminal_bonus * shrink

        self._action_history.append(a_in)

        info = {
            "dist_m": d_now,
            "h_err_m": h - self.dest_xyz[2],
            "violation": violation,
            "fuel_kg_step": f * self.cfg.dt_s,
        }
        return self._observe(), float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        """Pack the 21-vector observation."""
        from ..common.coordinates import enu_to_lla
        from ..common.wind_field import wind as _wind
        x, y, h, v, psi, m = self.state
        err = self.dest_xyz - np.array([x, y, h])
        m_frac = (m - self.ac.m_OEW_kg) / max(1.0, self.fp.initial.m_kg - self.ac.m_OEW_kg)
        t_frac = self._step_idx / max(1, self.cfg.max_steps)
        dist = float(np.linalg.norm(err[:2]))
        dist_norm = dist / max(1.0, self._init_dist)
        lam, phi = enu_to_lla(x, y, self.origin)
        Wx, Wy = _wind(float(lam), float(phi), self.wf)
        base = [
            err[0] / self._sx[0],
            err[1] / self._sx[1],
            err[2] / self._sh,
            h / self._sh,
            v / self._sv,
            float(np.sin(psi)),
            float(np.cos(psi)),
            float(np.clip(m_frac, 0.0, 1.0)),
            float(t_frac),
            float(dist_norm),
            float(Wx) / 50.0,
            float(Wy) / 50.0,
        ]
        # Append the action history oldest-first.  Each entry is the
        # raw [-1,1]^3 action the policy emitted at that step.
        hist = list(self._action_history)
        flat_hist = [float(v) for a in hist for v in a]
        return np.asarray(base + flat_hist, dtype=np.float32)

    def action_to_physical(self, a: np.ndarray) -> np.ndarray:
        """Public wrapper around the affine action map."""
        return self._to_physical(np.asarray(a, dtype=float))

    def physical_to_action(self, u: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`action_to_physical`."""
        span = self._u_hi - self._u_lo
        a = 2.0 * (np.asarray(u, dtype=float) - self._u_lo) / span - 1.0
        return np.clip(a, -0.999, 0.999).astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers used by `evaluate.py` and the BC/DAgger pipeline
    # ------------------------------------------------------------------
    def trajectory_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (t, state, control) for the last completed episode."""
        state = np.asarray(self.history_state)
        control = (np.asarray(self.history_control)
                   if self.history_control else np.zeros((0, 3)))
        n = state.shape[0]
        t = np.arange(n) * self.cfg.dt_s
        return t, state, control

    def observe_state(self, state: np.ndarray, step_idx: int,
                      action_history: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the observation that would be emitted at ``state``.

        Used by the BC dataset builder to resample expert trajectories
        without having to integrate them through the env. ``action_history``
        is the (k, 3) buffer of recent normalised actions ordered
        oldest-first; missing entries are zero-padded.
        """
        saved_state = self.state
        saved_idx = self._step_idx
        saved_hist = self._action_history.copy()
        self.state = np.asarray(state, dtype=float).copy()
        self._step_idx = int(step_idx)
        if action_history is not None:
            buf = np.asarray(action_history, dtype=np.float32)
            pad = ACTION_HISTORY_LEN - buf.shape[0]
            if pad > 0:
                buf = np.vstack([np.zeros((pad, 3), dtype=np.float32), buf])
            self._action_history = deque(
                [buf[i] for i in range(-ACTION_HISTORY_LEN, 0)],
                maxlen=ACTION_HISTORY_LEN,
            )
        obs = self._observe()
        self.state = saved_state
        self._step_idx = saved_idx
        self._action_history = saved_hist
        return obs

    def make_factory(self, seed: int = 0):
        """Return a callable that creates a fresh env (for VecEnv)."""
        flight_id = self.flight_id
        cfg = self.cfg

        def _f() -> "FlightPlannerEnv":
            return FlightPlannerEnv(flight_id=flight_id, cfg=cfg, seed=seed)
        return _f
