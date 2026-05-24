---
name: recompile-report
description: Compile report/report.tex into report/report.pdf (pdflatex twice for cross-references). Use after editing the report or refreshing result figures.
allowed-tools: Bash, Read
---

# recompile-report

Two-pass `pdflatex` build with bibtex in the middle so cross-references
(`\ref`, `\cite`) resolve. Run this every time `report.tex` is edited or
the `results/compare/` figures change.

## Steps

1. Verify the LaTeX source exists:
   ```bash
   test -f report/report.tex || { echo "ERROR: report/report.tex missing"; exit 1; }
   ```

2. First pass:
   ```bash
   cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex 2>&1 | tail -20
   ```

3. Bibtex (only if `\bibliography{...}` is used; current report uses
   inline `thebibliography`, so this is a no-op but kept for forward
   compatibility):
   ```bash
   cd report && (bibtex report 2>&1 | tail -10 || true)
   ```

4. Second pass to resolve cross-references and citations:
   ```bash
   cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex 2>&1 | tail -10
   ```

5. Confirm the PDF was regenerated:
   ```bash
   ls -lh report/report.pdf
   ```

## Output

A one-line PASS/FAIL with the page count and file size. If pdflatex
returns a non-zero exit, show the last 30 lines of the log so the user
can locate the error.
