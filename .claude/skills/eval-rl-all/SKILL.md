---
name: eval-rl-all
description: Roll out trained RL policies for all 3 flights, regenerate the 8 per-flight plots, build cross-method overlays, and print a one-page summary table. Use after the RL pipeline finishes.
allowed-tools: Bash, Read
---

# eval-rl-all

Glue script for the post-training pipeline. Assumes `models/rl/flight_<id>/policy.zip`
exists (produced by `src.rl.pipeline`) and that the direct-collocation
results are already on disk (`results/direct/flight_<id>/`).

## Steps

1. Roll out the saved policy for each flight; this writes
   `results/rl/flight_<id>/{solution.npz, meta.json}` plus the 8
   standard PNGs:
   ```bash
   python -m src.rl.evaluate --flight all
   ```

2. Build the direct-vs-RL overlay figures and the comparison table:
   ```bash
   python -m src.compare.make_report_figs
   ```

3. Print the cross-method comparison table:
   ```bash
   cat results/compare/summary.md
   ```

4. Pull the per-flight fuel/closest-distance line from each RL
   `meta.json` so the user sees the headline numbers at a glance:
   ```bash
   python - <<'PY'
   import json, pathlib
   for fid in (1, 2, 3):
       direct = json.loads(pathlib.Path(f"results/direct/flight_{fid}/meta.json").read_text())
       rl     = json.loads(pathlib.Path(f"results/rl/flight_{fid}/meta.json").read_text())
       closest_km = rl.get("solver_meta", {}).get("closest_dist_m", 0) / 1000.0
       print(f"F{fid}: DC fuel={direct['fuel_total_kg']:.0f} kg, "
             f"RL fuel={rl['fuel_total_kg']:.0f} kg, "
             f"RL closest={closest_km:.2f} km")
   PY
   ```

## Output

A clean PASS/FAIL line per stage plus the comparison table. If `evaluate`
fails for any flight, stop and surface the traceback -- do not proceed
to overlays.
