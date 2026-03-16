# Reproducibility Checklist (2026-03-16)

This note documents what is required to reproduce the current best model
results from this repository in the future.

## Goal

Reproduce inference and evaluation results for the current default checkpoint:

- `models/drum_classifier_20260315_223300_best.pth`

This means reproducing the same loaded-model behavior, not retraining from
scratch to bit-identical weights.

## What Git Already Preserves

- Project code
- Config defaults in `config.py`
- Changelogs and evaluation notes under `docs/results/`
- The viewer/data-loading fix that avoids rewriting the external processed MIDI CSV

## What Git Does Not Preserve By Itself

- Model checkpoint files under `models/`
- Source audio and MIDI files outside this repo
- The exact `drum310` Conda environment
- The exact training run state needed to regenerate the same weights from scratch

## Required Artifacts

To reproduce the current best results, keep a copy of all of the following:

### Repository

- Git repo at commit `8bdcaeb` or later on `master`

### Checkpoints

- `models/drum_classifier_20260315_223300_best.pth`
- Optional comparison checkpoints:
  - `models/drum_classifier_20260315_223300_best_best_f1.pth`
  - `models/drum_classifier_20260315_210518_best.pth`

### External data

The current config expects the original source tree:

- `C:\Users\windo\OneDrive\Python\drums_to_midi\00_GM`

Required files under that tree:

- `midi\reaper_GM_audio_1.wav`
- or fallback `midi\reaper_GM_audio.wav`
- `midi\reaper_midi_GM.csv`

## Required Config State

These paths in `config.py` must resolve correctly on the target machine:

- `BASE_DIR`
- `SOURCE_DIR`
- `AUDIO_FILE`
- `FALLBACK_AUDIO_FILE`
- `MIDI_CSV_INPUT`
- `MODEL_LOAD_PATH`

Current intended default:

- `MODEL_LOAD_PATH = models/drum_classifier_20260315_223300_best.pth`

If the repo is restored to a different location or the source data lives
elsewhere, update `config.py` accordingly.

## Environment Capture

To make restoration easier, export the current Conda environment on the source
machine and keep the export with the project archive.

Recommended commands:

```powershell
conda activate drum310
conda env export -n drum310 > environment_drum310.yml
pip freeze > requirements_locked.txt
```

`requirements.txt` in this repo is intentionally loose, so the exported Conda
environment is the more important record.

## Restore Procedure

On a future machine:

1. Clone the repo.
2. Restore the checkpoint file into `models\`.
3. Restore the external source data and update `config.py` paths if needed.
4. Recreate the environment from the saved export if available.
5. Run a small sanity check.

Example:

```powershell
conda env create -f environment_drum310.yml
conda activate drum310
python src\infer.py 9145
```

## Sanity Checks

Use these checks after restoration:

```powershell
python -c "import config; print(config.MODEL_LOAD_PATH); print(config.get_audio_file())"
python src\infer.py 9145
python src\view.py
```

Expected project facts from the current repo state:

- Dataset size should build to `13680` slices
- The current best checkpoint should load with the current `src/model.py`
- `src/view.py` should work without regenerating the external processed MIDI CSV

## Limits

This project is reproducible for inference if the checkpoint, data, and
environment are preserved.

It is not fully reproducible for exact retraining from scratch unless you also
preserve:

- the exact package versions
- the exact source data files
- the exact checkpoint/resume state
- the exact random state and runtime environment

## Recommendation

For durable recreation, archive these together:

- repo checkout
- `models\drum_classifier_20260315_223300_best.pth`
- external source data snapshot
- `environment_drum310.yml`
- `requirements_locked.txt`
