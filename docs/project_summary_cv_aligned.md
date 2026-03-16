# Drum Audio to MIDI Project Summary

## Executive Framing

This project is a defined-scope AI systems initiative focused on converting raw
recorded drum audio into structured MIDI events with usable note, timing, and
velocity output. It reflects end-to-end ownership of the machine learning
lifecycle: problem definition, dataset creation, model architecture, training
pipeline design, inference tooling, and validation.

The current repository is the release-oriented Python refactor of the original
working notebook. Its purpose is not broad experimentation. The goal is to
preserve a stable, reproducible baseline that can be evaluated, trained, and run
through a controlled CLI workflow.

## What Was Built

- An end-to-end audio-to-MIDI pipeline in Python using PyTorch
- A trainable STFT front end feeding a 1D encoder/decoder convolutional network
- Data preparation logic that converts aligned audio and MIDI source material
  into fixed 2.5-second supervised training slices
- Training, checkpointing, evaluation, and inference paths that support
  repeatable validation against held-out data
- Interactive and CLI-based tools for reviewing predicted drum events against
  source targets

## Technical Scope

The system takes WAV audio, transforms it into a learned spectral
representation, predicts a MIDI-style heatmap, and extracts drum events from
that output. The implementation includes:

- Dataset engineering for time-aligned audio and MIDI supervision
- Model design under practical compute constraints
- Checkpointed training with preserved best-loss and best-F1 artifacts
- Contiguous holdout evaluation to reduce leakage risk
- Inference-time thresholding and optional non-maximum suppression
- Reproducibility guidance covering code, checkpoints, environment, and
  external source data dependencies
- A limited dataset scope centered primarily on a known drum kit, with broader
  generalization requiring more diverse source data

## Current Repository Status

This codebase is positioned as a controlled baseline rather than an open-ended
research branch. It currently preserves:

- The original notebook-era model behavior
- Centralized configuration in `config.py`
- Timestamped model saves for new training runs
- Resume support through `models/checkpoint_resume.pth`
- Sanity-checked inference compatibility with the current best checkpoint
- Explicit dependency on local training data and checkpoints referenced through
  `config.py` but not included in the repository

Recent project notes indicate:

- Dataset build size: `13680` slices
- Current preferred checkpoint:
  `models/drum_classifier_20260315_223300_best.pth`
- Verified workflow coverage in the `drum310` environment for config import,
  model instantiation, dataset build, checkpoint load, and single-slice
  inference

## Resume-Aligned Positioning

In CV terms, this project demonstrates:

- AI feasibility execution from concept to validated implementation
- ML architecture ownership across data, model, training, and inference
- Practical engineering discipline under constrained local compute conditions
- Architectural control focused on reproducibility, failure avoidance, and
  measurable output quality
- Translation of a working research prototype into a more stable,
  release-oriented software asset

## Concise Summary

Developed an end-to-end drum audio-to-MIDI AI pipeline in Python/PyTorch,
covering dataset engineering, model architecture, training workflow,
checkpointed evaluation, and inference tooling. Refactored the original working
notebook into a controlled baseline repository designed for reproducible
validation and operational use rather than unconstrained experimentation. The
working dataset and checkpoints remain available locally through `config.py`,
but they are not part of the repository, and the current dataset is not broad
enough to support strong generalization claims beyond the known drum kit
distribution.
