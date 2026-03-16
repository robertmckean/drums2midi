# CHANGELOG v0.1.7

Add focused explanatory comments across the core Python pipeline files.

## Summary

This release improves code readability for external reviewers by documenting the
non-obvious logic in training, data preparation, model execution, inference,
interactive viewing, and MIDI alignment.

## Changes

### Code Comments
- Updated `src/train.py` with clearer comments on checkpoint resume behavior,
  metric calculation, and best-model save semantics
- Updated `src/data.py` with comments on MIDI heatmap construction and the
  contiguous train/test gap rationale
- Updated `src/model.py` with comments on STFT cropping and encoder/decoder flow
- Updated `src/infer.py` with comments on threshold units, NMS grouping, and
  dataset rebuilding
- Updated `src/view.py` with comments on viewer dataset usage and console/GUI flow
- Updated `src/robert.py` with comments on MIDI time scaling and returned slice structure

## Notes

- This is a readability/documentation pass only; no model or pipeline behavior
  was intentionally changed
