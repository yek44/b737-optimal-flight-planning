---
name: validate-code
description: Runs flake8 + mypy + pytest in sequence. Use before commits or when a code-quality gate is needed.
allowed-tools: Bash
---

# validate-code

Runs the full local quality gate.

## Steps

1. Lint with flake8:
   ```bash
   flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503 --statistics
   ```
2. Type check with mypy:
   ```bash
   mypy src/ --ignore-missing-imports
   ```
3. Test with pytest:
   ```bash
   pytest tests/ -v --tb=short
   ```

## Output
Report PASS/FAIL per stage and a one-line summary. If any stage fails, do NOT proceed
to the next stage; surface the first failure and stop.
