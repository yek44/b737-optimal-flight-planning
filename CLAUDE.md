# UCK 439E - Optimal Control Term Project (B737-800 Wind-Field Flight Planning)

## What this project is
Single-student Operations Research / Optimal Control term project. Course UCK 439E (ITU). Due 2026-06-07.

Goal: optimize B737-800 cruise-phase trajectories under polynomial wind fields for 3 prescribed flights, using:
- (a) Direct collocation (CasADi + IPOPT, Hermite-Simpson)
- (b) Reinforcement Learning (Stable-Baselines3 PPO + BC warm-start + DAgger expert correction)

Detailed problem spec: see `Term Project.pdf` in repo root.

Note: An earlier reduced-state DP module was implemented and later dropped — the 2-D
cruise model lost too much realism (fixed altitude/airspeed/throttle) to be a meaningful
comparison against the 6-state direct and RL solutions.

## Tech stack (locked)
- Python 3.11
- CasADi + IPOPT/MUMPS (precompiled wheel) - direct collocation
- numpy, scipy, matplotlib - common utilities + plotting
- gymnasium + stable-baselines3 + torch (CPU) - RL
- pyyaml - config files
- pytest, flake8, mypy - quality gates

## Build & test commands (Windows Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate           # Git Bash
# OR  .venv\Scripts\activate.bat        # cmd.exe
pip install --upgrade pip
pip install -r requirements.txt
pytest tests/ -v
```

## Repo layout
```
src/common/      atmosphere, wind_field, aircraft, dynamics, coordinates, cost, plotting, results
src/direct/      Hermite-Simpson collocation (CasADi)
src/rl/          Gymnasium env + BC + DAgger + SB3 PPO training
src/compare/     overlay figures for the report
data/            flight_plans.yaml, aircraft_b738.yaml, wind_field.yaml
tests/           pytest suites for common modules
notebooks/       sanity checks + result exploration
results/         canonical outputs (one set per flight per method)
report/          LaTeX source + final report.pdf
```

## Math / physics conventions (NON-NEGOTIABLE)
- **Units**: SI internally (meters, m/s, kg, radians, seconds). Degrees and kg-fuel only at I/O boundaries (config files, plots, report).
- **Coordinates**: integrate dynamics in flat-Earth ENU meters with per-flight reference origin `(lambda0, phi0)`. Convert ENU back to (lambda deg, phi deg) only when (i) querying the wind polynomial Wx/Wy and (ii) producing plots.
- **Angles**: radians internally always. Use `numpy.unwrap` on `psi` before plotting.
- **Heading `psi` is measured CCW from east (math convention)**, NOT clockwise from north (aviation). The PDF equations `dx = v*cos(psi)`, `dy = v*sin(psi)` force this. So `psi=0` -> heading east; `psi=pi/2` -> heading north. Table 3's `psi_0 = 0 deg` means initial heading due east.
- **Mass monotonicity**: assert `dm/dt <= 0` everywhere; mass cannot grow.
- **Free final time**: `tf` is a decision variable; integrate on normalized `tau in [0,1]`, scale derivatives by `tf`.

## Coding standards
- PEP 8, max line length 100.
- Type hints required on every public function.
- Google-style docstrings on every public function/class.
- No comments that just restate the code.
- No fallbacks for impossible cases (raise; don't swallow).
- No premature abstractions; YAGNI.
- No stubbed functions left behind - finish or delete.

## Optimization-specific guardrails

### CasADi NLP
- Always nondimensionalize states before passing to IPOPT (scales: x,y/1e5; h/1e4; v/250; m/1e5; t/3600).
- Always supply an initial guess from the great-circle baseline (`src/direct/initial_guess.py`).
- Wrap `psi` carefully; consider `(sin psi, cos psi)` decomposition if wraparound bites you.

### IPOPT
- Linear solver: MUMPS (default Windows wheel; HSL not available without a license).
- Defaults: `tol=1e-6`, `mu_strategy='adaptive'`, `hessian_approximation='limited-memory'`, `max_iter=3000`.
- Warm-start flight 2 and 3 from flight 1's solution.

### Gymnasium / SB3
- Always wrap `SubprocVecEnv` instantiation in `if __name__ == "__main__":` (Windows multiprocessing requirement).
- Wrap env with `VecNormalize` for obs+reward normalization.
- Save and reload `VecNormalize` stats with the policy when evaluating.

### Output hygiene
- Never write `gurobi.log`, `*.lp`, `*.mps` to the repo (gitignored anyway).
- Plot policy: every figure has axis labels, units in brackets `[m]`, `[m/s]`, `[deg]`, `[s]`, and a legend. No bare titles.

## Testing rule
Any change touching `src/common/dynamics.py` or `src/common/wind_field.py` must rerun `tests/test_dynamics.py` and `tests/test_wind_field.py` before commit. The `auto-test-dynamics.sh` PostToolUse hook automates this; do not rely on memory.

## Subagents available
- `math-verifier` - verifies Python implementations against the PDF equations (sign conventions, units, edge cases). Invoke after editing dynamics/wind/cost.
- `plot-reviewer` - checks generated PNGs for axis labels, units, legends, sensible y-ranges.
- `report-reviewer` - pre-submission audit of `report/report.tex` (placeholders, broken refs/cites, missing figures, numerical inconsistencies between table and inline text). Invoke right before recompiling the final `report.pdf`.

## Slash commands / skills
- `/validate-code` - flake8 + mypy + pytest.
- `/run-flight` - wraps `python -m src.<method>.solve --flight {1,2,3}` and dumps a one-page summary.
- `/eval-rl-all` - roll out trained RL policies for all 3 flights, regenerate the 8 per-flight plots, build cross-method overlays, and print a one-page summary. Use after `src.rl.pipeline` finishes.
- `/recompile-report` - two-pass pdflatex on `report/report.tex`. Use after the report or any `results/compare/` figure changes.

## Hooks (auto-fired by the harness)
- `lint-on-edit.sh` (PostToolUse on Edit|Write) - non-blocking flake8 on edited .py files.
- `auto-test-dynamics.sh` (PostToolUse on Edit|Write) - reruns `tests/test_dynamics.py` and `tests/test_wind_field.py` whenever `src/common/{dynamics,wind_field,aircraft,cost}.py` is edited.
- `block-result-overwrites.sh` (PreToolUse on Edit|Write) - hard-blocks manual edits inside `results/{direct,rl,compare}/flight_*` so solver artefacts are only written by the responsible pipelines.
- `notify-done.sh` (Stop) - bell + console banner when Claude finishes.

## Model strategy
- Opus for: algorithm design (Hermite-Simpson formulation, RL reward shaping, DAgger expert correction), debugging IPOPT non-convergence, report structuring.
- Sonnet (default) for: implementation, tests, plotting, edits.
- Haiku for: trivial renames, doc tweaks, plot-reviewer subagent.
- Plan mode at the start of every non-trivial session.
- `ultrathink` only when stuck on math derivations or IPOPT convergence.

## Git workflow
- One branch per method: `method/direct`, `method/rl`. Merge to `main` when working on Flight 1.
- Conventional-ish commits: `feat(direct): hermite-simpson collocation builder`.
- Never `git push --force`; never commit `.venv/`, `results/*.npz` larger than 1 MB, `__pycache__/`.

## Submission
Single `<studentID>.rar` per the PDF. Include `report.pdf` + source. Exclude `.venv`, large binaries, notebook checkpoints.
