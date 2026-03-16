# CHANGELOG v0.1.4

Document the exact requirements needed to reproduce the current best model
results in the future.

## Summary

This release adds a reproducibility checklist covering the current best
checkpoint, required external data files, required config state, and recommended
environment exports.

## Changes

### Documentation
- Updated `README.md` to point at the current default checkpoint
- Added `docs/results/reproducibility_20260316.md`
- Documented the exact files and restore steps needed to reproduce current results

## Notes

- Git preserves the code and docs, but not the model checkpoint files or the
  external source data by itself
- Exact retraining from scratch is still not guaranteed without preserving the
  environment, data, and training state
