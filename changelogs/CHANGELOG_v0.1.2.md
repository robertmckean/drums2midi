# CHANGELOG v0.1.2

Publish checkpoint comparison results for the contiguous-holdout training run.

## Summary

This release adds a documented checkpoint comparison showing that the newer
contiguous-holdout run outperformed the older random-split baseline on a
250-slice sampled evaluation.

## Changes

### Results
- Added `docs/results/checkpoint_comparison_20260316.md`
- Recorded aggregate F1 for three matching modes: exact, note+time, and tolerant
- Documented the ranking of the baseline, best-loss, and best-F1 checkpoints
- Included pairwise per-slice win counts for the 250-slice sample

## Validation

- Evaluation dataset size: `13680` slices
- Comparison sample size: `250` slices with random seed `42`
- Top checkpoint on `note_time` F1: `models/drum_classifier_20260315_223300_best.pth`
- `note_time` F1 scores:
  - `210518_best`: `0.9221`
  - `223300_best`: `0.9391`
  - `223300_best_best_f1`: `0.9373`

## Notes

- This changelog records docs/results only; no training or inference code changed
- Existing local working-tree code edits were intentionally left out of this release
