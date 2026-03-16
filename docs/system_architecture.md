# System Architecture

## Overview

`drums2midi` is a controlled baseline AI pipeline that converts recorded drum
audio into structured MIDI-style event predictions. The repository preserves
the implementation path from aligned source data through training, inference,
and visual review.

## Pipeline

```text
Recorded drum audio (WAV)
-> audio / MIDI alignment
-> fixed 2.5-second supervised slices
-> trainable STFT front end
-> 1D convolutional encoder / decoder
-> MIDI heatmap prediction
-> thresholding and optional NMS
-> note / time / velocity event extraction
-> visual review and comparison
```

## Component Map

- `config.py`
  Centralizes paths, hyperparameters, checkpoint selection, and runtime
  defaults.
- `src/data.py`
  Builds the supervised dataset from aligned audio and MIDI sources, slices the
  waveform into fixed windows, and prepares dataloaders.
- `src/robert.py`
  Handles MIDI CSV slicing and alignment logic used by the data pipeline.
- `src/model.py`
  Defines the baseline model: trainable STFT front end, 1D convolutional
  encoder, and transposed-convolution decoder producing a MIDI-style heatmap.
- `src/train.py`
  Runs training, checkpointing, test evaluation, and plot generation.
- `src/infer.py`
  Loads the configured checkpoint, runs slice-level inference, applies output
  post-processing, and saves comparison plots.
- `src/view.py`
  Provides an interactive slice viewer for qualitative review of predictions
  against target events.

## Model Notes

The current baseline preserves the original notebook-era architecture rather
than introducing a broader redesign. The active model characteristics are:

- trainable STFT front end
- 1D convolutional encoder
- 1D transposed-convolution decoder
- raw output head without sigmoid

This keeps the repository anchored to the validated baseline rather than an
open-ended experimental branch.

## Training And Evaluation

Training uses supervised audio/MIDI pairs prepared from the local source tree
referenced in `config.py`. The project currently preserves:

- timestamped model saves for new best checkpoints
- resumable training state
- saved training loss and F1 plots
- contiguous holdout evaluation with a configurable gap to reduce leakage risk

Recent comparison notes indicate that the current preferred checkpoint is:

- `models/drum_classifier_20260315_223300_best.pth`

See `docs/results/checkpoint_comparison_20260316.md` for the documented
comparison against the older random-split baseline.

## Data And Reproducibility Boundaries

This repository is not self-contained from Git alone.

- training data and checkpoints exist locally and are referenced through
  `config.py`
- those assets are not included in the repository
- the current dataset is limited and primarily represents a known drum kit
- broader generalization would require a more diverse dataset

See `docs/results/reproducibility_20260316.md` for the current restore and
validation checklist.

## Why This Structure Matters

For technical reviewers, the repository shows a complete implementation path:
data preparation, model definition, training loop, inference path, and review
tooling.

For executive reviewers, it shows a defined-scope AI system executed with
architectural discipline: constrained baseline control, explicit checkpoint
management, documented validation, and realistic handling of reproducibility and
generalization limits.
