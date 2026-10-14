# Step 8 — Submission report

**Code:** [`src/report.py`](../report.py) · **Output:** [`src/freight-rate-report/freight-rate-report.pdf`](../freight-rate-report/freight-rate-report.pdf)

```bash
poetry install --no-root --with report   # reportlab, in its own dependency group
poetry run python -m src.report
```

## What the brief asks for

A PDF or DOCX report containing:

1. The approach to validating and splitting the train / test data.
2. The fixed December prediction chart produced by `score.py`.

## Contents (3 pages)

| Section | Content |
|---|---|
| Summary | The out-of-time principle and the headline numbers |
| 1. Data split | Why split by month; train / validation / test table; the test-set rule; Figure 1, a timeline of every evaluation round |
| 2. Validation approach | Expanding-window CV, the metric, leakage control, corrupted labels (removed from training, kept in evaluation), the selection rule |
| 3. Results | CV table and Figure 2 (MAPE per month for all four models); October test table and how to read it |
| 4. December chart | `scorer_results/candidate_december.png` with its level, shape and holiday limitation |
| Reproducing the results | The full command sequence |

## Design choices

- **Every number comes from the result files** written by the earlier steps (`src/outputs/models/`, `tuning/`, `test/`, and the filled December file), so the report cannot drift from the code's output.
- **Two figures made for the report** go to `src/outputs/report/`: `split_timeline.png` and `mape_by_month.png` (the Step 5 chart did not include the final model).
- **PDF via reportlab:** no document converter (LibreOffice, pandoc) is required on the machine. reportlab lives in a separate `report` dependency group, so the modeling environment stays lean.
- **Helvetica only**, so the PDF renders the same on any machine.
