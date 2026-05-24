---
name: run-flight
description: Runs a single flight through one method (direct or rl) and prints a one-page summary (cost, fuel, duration, convergence). Use when iterating on a method.
allowed-tools: Bash, Read
---

# run-flight

Runs `python -m src.<method>.solve --flight <n>` and summarizes the result.

## Arguments (parse from $ARGUMENTS or ask the user)
- `method`: one of `direct`, `rl`
- `flight`: 1, 2, or 3

## Steps

1. Run the solver:
   ```bash
   python -m src.{method}.solve --flight {flight}
   ```
2. Read the resulting `results/{method}/flight_{flight}/meta.json`.
3. Print a one-page summary:
   - Method, flight ID
   - Converged: yes/no
   - Objective cost
   - Total fuel [kg]
   - Total duration [s]
   - Solver wall time [s]
   - First 3 lines of solver-specific output (e.g. IPOPT iter count)

## Output
A clean human-readable report. Do not regenerate plots; only summarize.
