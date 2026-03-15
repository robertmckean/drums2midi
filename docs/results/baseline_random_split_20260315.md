# Baseline Random-Split Results (2026-03-15)

This note records the final baseline results from the random slice split before
switching training to the stricter contiguous holdout split with a gap.

## Model

- Checkpoint: `models/drum_classifier_20260315_210518_best.pth`
- Training stopped manually after epoch 47

## Final Logged Training Snapshot

- Epoch 47 train loss: `4.0177e-06`
- Epoch 47 test loss: `1.5215e-05`
- Epoch 47 train F1: `0.8987`
- Epoch 47 test F1: `0.8130`
- Epoch 47 train precision / recall: `0.86 / 0.942`
- Epoch 47 test precision / recall: `0.75 / 0.882`

## Saved Comparison Artifacts

- Raw training log: `results/drum_classifier_20260315_210518_best.txt`
- Random 10-slice review: `docs/results/random_slice_review_20260315_222716.txt`
- Inference plots: `files/inference_slice_*.png`

## Notes

- The 10-slice review used random seed `42`
- Selected slices: `409, 1679, 1824, 2286, 3657, 4012, 4506, 10476, 12066, 12149`
- Average exact note/time per-slice F1 in that review: `0.9554`
- These results are useful for comparison, but they come from the older random
  slice split and may be optimistic because train and test slices came from the
  same long recording
