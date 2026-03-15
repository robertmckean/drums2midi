# CHANGELOG v0.1.1

Tighten evaluation by replacing the random slice split with a contiguous
holdout split and add separate best-F1 checkpoint saves during training.

## Summary

This release makes the train/test evaluation path more trustworthy for the
single long source recording. Training no longer mixes random slices from
across the file into both train and test sets. It now uses a contiguous
holdout split with a configurable gap, and it preserves both the best-loss and
best-F1 checkpoints for later comparison.

## Changes

### Data split
- Replaced the random slice split in `src/data.py` with a contiguous split
- Added `TRAIN_TEST_GAP_MINUTES` in `config.py`
- Disabled test-loader shuffle so evaluation order stays fixed
- Added split summary logging that prints the train range, gap, and test range

### Training checkpoints
- Added a separate best-F1 checkpoint save alongside the best-loss checkpoint
- Extended checkpoint resume metadata to persist best-F1 state and its save path
- Added final training summary output for best test F1

### Repo and docs
- Updated `README.md` to describe the contiguous holdout evaluation and dual
  checkpoint saves
- Narrowed the root `results/` ignore rule so `docs/results/` can be tracked

## Validation

- Verified the new split path in `drum310`
- Dataset size: `13680`
- Train/test split summary: `train slices 0-10751`, `gap 240 slices`,
  `test slices 10992-13679`
- DataLoader sizes: `672` train batches, `168` test batches

## Notes

- The current baseline comparison notes remain under `docs/results/`
- Existing generated model files and training plots were not committed in this release
