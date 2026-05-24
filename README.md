# UCK 439E Term Project - B737-800 Wind-Field Flight Planning

Optimized cruise-phase trajectories for a Boeing 737-800 under a polynomial wind
field, solved two different ways for comparison:

1. **Direct collocation** - CasADi + IPOPT, Hermite-Simpson (benchmark)
2. **Reinforcement Learning** - Stable-Baselines3 PPO + behavioural-cloning warm
   start + DAgger-style expert correction

Course: UCK 439E (Introduction to Optimal Control), ITU
Due: 2026-06-07

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate     # Git Bash
# OR .venv\Scripts\activate.bat   # cmd.exe
pip install --upgrade pip
pip install -r requirements.txt
pytest tests/ -v
```

## Reproducing the report figures

```bash
# Direct collocation (3 flights, ~1 min CPU total)
python -m src.direct.solve --flight all --warm-start

# Reinforcement learning (~1-2 h CPU per flight on a single core)
python -m src.rl.pipeline --flight 1
python -m src.rl.pipeline --flight 2
python -m src.rl.pipeline --flight 3
python -m src.rl.evaluate --flight all

# Comparison figures + tables
python -m src.compare.make_report_figs
```

Outputs land in `results/<method>/flight_<n>/`. The 8 standard plots per flight
are named `xy.png`, `h_time.png`, `v_time.png`, `mass_time.png`, `thrust_time.png`,
`throttle_time.png`, `mu_time.png`, `gamma_time.png`.

## Repo layout
See `CLAUDE.md` for the full module map.

## Optional: Context7 MCP for live library docs

Context7 (Upstash) keeps Claude's view of CasADi / Gymnasium / Stable-Baselines3
APIs fresh. To enable:

```bash
claude mcp add context7 -- npx -y @upstash/context7-mcp
```

It works without an API key for low usage; sign up at upstash.com if you want
higher rate limits.

## Submission

Per the project PDF: zip everything except `.venv/` and large result binaries
into `<studentID>.rar` and upload to Ninova along with `report/report.pdf`.
