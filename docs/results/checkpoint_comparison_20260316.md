# Checkpoint Comparison (2026-03-16)

This note records a direct checkpoint comparison after moving to the stricter
contiguous holdout split in `v0.1.1`.

## Evaluation Setup

- Dataset: current full dataset built from `config.py`
- Dataset size: `13680` slices
- Sample size: `250` slices
- Slice selection: random sample with seed `42`
- Matching modes:
  - `exact`: note, time, and velocity must all match
  - `note_time`: note and time must match; velocity ignored
  - `tolerant`: note must match, time within `+-2` steps, velocity within `+-10`

## Compared Checkpoints

- `models/drum_classifier_20260315_210518_best.pth`
  - Older baseline from the pre-`v0.1.1` random-split training run
- `models/drum_classifier_20260315_223300_best.pth`
  - Best test-loss checkpoint from the newer contiguous-holdout training run
- `models/drum_classifier_20260315_223300_best_best_f1.pth`
  - Best test-F1 checkpoint from the newer contiguous-holdout training run

## Aggregate Results

| Checkpoint | Exact F1 | Note+Time F1 | Tolerant F1 |
| --- | ---: | ---: | ---: |
| `210518_best` | `0.0743` | `0.9221` | `0.7313` |
| `223300_best` | `0.1189` | `0.9391` | `0.7406` |
| `223300_best_best_f1` | `0.1106` | `0.9373` | `0.7279` |

## Ranking

1. `models/drum_classifier_20260315_223300_best.pth`
2. `models/drum_classifier_20260315_223300_best_best_f1.pth`
3. `models/drum_classifier_20260315_210518_best.pth`

For the most practical event metric here, `note_time`, the best-loss checkpoint
was strongest with precision `0.9467`, recall `0.9317`, and F1 `0.9391`.

## Pairwise Slice Wins

Per-slice wins are noisy, but they support the aggregate ranking:

- Exact:
  - `210518_best` vs `223300_best`: `67-120`, ties `63`
  - `210518_best` vs `223300_best_best_f1`: `74-115`, ties `61`
  - `223300_best` vs `223300_best_best_f1`: `84-63`, ties `103`
- Note+Time:
  - `210518_best` vs `223300_best`: `51-90`, ties `109`
  - `210518_best` vs `223300_best_best_f1`: `52-89`, ties `109`
  - `223300_best` vs `223300_best_best_f1`: `21-13`, ties `216`
- Tolerant:
  - `210518_best` vs `223300_best`: `56-140`, ties `54`
  - `210518_best` vs `223300_best_best_f1`: `59-135`, ties `56`
  - `223300_best` vs `223300_best_best_f1`: `52-11`, ties `187`

## Notes

- On this 250-slice sample, the newer contiguous-holdout run clearly outperformed
  the older baseline.
- The epoch-76 best-loss checkpoint beat the epoch-71 best-F1 checkpoint on all
  three evaluation modes in this comparison.
- The best-F1 training checkpoint still remains useful as a preserved alternate
  selection criterion, but it was not the strongest checkpoint on this sampled
  event-level evaluation.
