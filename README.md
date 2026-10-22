# Wind-Aware B737-800 Flight Planning

Optimized cruise-phase trajectories for a Boeing 737-800 in a
polynomial wind field. The repository compares an exact-endpoint direct
collocation solver with a route-conditioned PPO flight-planning policy
on the same six-state aircraft model.

| Method | Algorithm | Solver | Terminal handling |
|---|---|---|---|
| Direct collocation | Hermite-Simpson transcription, free final time, 241-node mesh | CasADi + IPOPT/MUMPS | Hard endpoint equality |
| Route-conditioned PPO | Sampled origin-destination training with continuous route context | Stable-Baselines3 PPO | General planner benchmarked on the three prescribed routes |
| Benchmark PPO | PPO actor initialized from expert-labelled trajectory data and refined on policy | Stable-Baselines3 PPO | Fuel/time comparison on the nominal routes |

The route-conditioned policy is the general flight-planner artifact. It
samples synthetic routes inside the wind-field domain and observes
route-relative features instead of a hard-coded Flight 1/2/3 label. The
current pilot reached 3/3 prescribed benchmark flights and 10/10
held-out sampled routes.

## Numerical Results

Unified PPO policy versus direct collocation on the three benchmark flights
(same numbers as `results/compare/summary.md` and the report):

| Flight | Route | Direct fuel | PPO fuel | Fuel | Direct time | PPO time | Time |
|:---:|:---|---:|---:|---:|---:|---:|---:|
| F1 | East-northeast long route | 8 560 kg | 8 647 kg | +1.0 % | 225.7 min | 228.8 min | +1.4 % |
| F2 | Southwest short route with sharp turn | 2 158 kg | 2 189 kg | +1.4 % | 71.3 min | 74.7 min | +4.8 % |
| F3 | Northbound route with terminal descent | 4 114 kg | 4 146 kg | +0.8 % | 117.0 min | 122.8 min | +5.0 % |

Direct collocation enforces the terminal waypoint exactly. The PPO
policy is evaluated by entry into a 500 m horizontal / 300 m vertical
terminal disk, so fuel and duration are the most meaningful cross-method
comparison metrics.

The final report is [`report/report.pdf`](report/report.pdf).

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate     # Git Bash on Windows
# OR  .venv\Scripts\activate.bat  # cmd.exe
pip install --upgrade pip
pip install -r requirements.txt
pytest tests/ -v
```

## Reproducing the Report

```bash
# Direct collocation
python -m src.direct.solve --flight all --warm-start

# Unified RL policy (BC + DAgger + PPO)
python -m src.rl.pipeline_unified --n-perturb 6 --bc-epochs 400 --dagger-rounds 4 --dagger-epochs 120 --ppo-steps 600000

# Evaluation and report figures
python -m src.rl.evaluate --flight all
python -m src.compare.make_report_figs
cd report && pdflatex report.tex && pdflatex report.tex
```

Outputs are written under `results/<method>/flight_<n>/`. Each flight
has the eight standard plots: `xy.png`, `h_time.png`, `v_time.png`,
`mass_time.png`, `thrust_time.png`, `throttle_time.png`, `mu_time.png`,
and `gamma_time.png`.

## Repository Layout

```text
src/common/      atmosphere, wind field, aircraft, dynamics, coordinates, cost, plotting
src/direct/      Hermite-Simpson direct collocation
src/rl/          Gymnasium environment, PPO training, policy evaluation
src/compare/     cross-method overlay figures and summary tables
data/            flight plans, aircraft parameters, wind-field coefficients
tests/           pytest suites
report/          LaTeX source and final PDF
results/         per-method per-flight outputs
```

## Quality Gates

```bash
flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
mypy src/ --ignore-missing-imports
pytest tests/ -v
```

## Author

Yunus Emre Kara, Istanbul Technical University.
