# Drum Audio to MIDI

This workspace is the release-oriented Python version of the original working
notebook `notebooks/DrumMidi_Working_LOCAL_00_GM.ipynb`.

The goal of this project is narrow: take recorded drum audio, convert it into a
MIDI-style heatmap, and extract note / time / velocity events with the original
baseline model behavior preserved. This workspace is intended to be a stable
finishing point, not an ongoing experimental branch.

## What This Version Is

This project keeps the original notebook-era model architecture and core training
behavior, while preserving a few practical improvements from the later refactor:

- central `config.py` path and hyperparameter management
- timestamped model saves for new training runs
- resumable training checkpoint support
- contiguous holdout train/test split with a configurable gap
- CLI inference and interactive viewing scripts
- saved loss and F1 plots
- separate best-loss and best-F1 checkpoint saves during training

## Baseline Model

The active checkpoint in this workspace is:

- `models/drum_classifier_20260315_223300_best.pth`

This checkpoint matches the current `src/model.py` architecture in this
workspace. The model is the original narrower encoder/decoder CNN with:

- trainable STFT front end
- 1D convolutional encoder
- 1D transposed-convolution decoder
- raw output head with no sigmoid

## Pipeline

```text
WAV audio
-> fixed 2.5 second slices
-> trainable STFT
-> 1D convolutional encoder/decoder
-> MIDI heatmap prediction
-> thresholding
-> optional NMS in inference
-> note / time / velocity events
```

## Repository Structure

- `config.py`: paths and hyperparameters
- `src/data.py`: audio slicing, MIDI heatmap preparation, dataloaders
- `src/robert.py`: MIDI CSV slicing and alignment logic
- `src/model.py`: original notebook baseline model
- `src/train.py`: training loop, checkpointing, loss/F1 plots
- `src/infer.py`: CLI inference and saved comparison plots
- `src/view.py`: interactive slice viewer
- `notebooks/`: original working notebook preserved for project provenance
- `examples/`: curated sample training and inference visuals for quick review
- `models/`: saved checkpoints
- `files/`: generated plots and inference images
- `changelogs/`: historical notes copied from the earlier repo

## Examples

Representative visuals are included under `examples/` for fast review without
requiring local model or data setup. These are curated copies of training and
inference outputs from the local working environment.

## Documentation

Key supporting documents:

- `docs/project_summary_master.md`: concise canonical project summary
- `docs/system_architecture.md`: pipeline and component overview
- `docs/model_detail.md`: detailed model architecture, tensor shapes, and parameter counts
- `docs/results/checkpoint_comparison_20260316.md`: checkpoint comparison notes
- `docs/results/reproducibility_20260316.md`: restore and reproducibility checklist
- `docs/presentation/drums2midi_Overview.pdf`: presentation walkthrough for mixed technical and executive audiences

## Paths And Data

This workspace stores code under:

- `C:\Users\windo\VS_Code\drums2midi`

Source audio and MIDI data are still read from the original local data folder
configured in `config.py`:

- `C:\Users\windo\OneDrive\Python\drums_to_midi\00_GM`

Important:

- training data and checkpoints exist locally, but are not included in this
  repository
- this project is not self-contained without those local source files
- `config.py` must point to valid audio and MIDI paths on the machine running it
- `src/data.py` uses `reaper_GM_audio_1.wav` first and falls back to `reaper_GM_audio.wav`
- the current dataset is limited and primarily represents a known drum kit
- broader generalization would require a more diverse dataset

## Environment

Use the `drum310` Conda environment for Python commands in this project.

Example:

```powershell
conda activate drum310
```

Install requirements if needed:

```powershell
pip install -r requirements.txt
```

## Reproducibility

Git alone is not enough to reproduce the current best results. To recreate the
same inference behavior in the future, you need:

- this repository at the current commit
- the external source data folder configured in `config.py`
- the saved checkpoint file referenced by `MODEL_LOAD_PATH`
- a compatible Python environment, ideally matching `drum310`

This repository preserves the implementation and current validated baseline, but
not the full local operating context by itself. The local dataset is usable for
the current project scope, but it is not broad enough to support strong claims
about generalization beyond the known training distribution.

See `docs/results/reproducibility_20260316.md` for the exact checklist and
restore steps.

For a concise pipeline walkthrough, see `docs/system_architecture.md`.
For detailed model architecture and parameter information, see
`docs/model_detail.md`.

## Running

Train:

```powershell
python src\train.py
```

Run inference on the default slice:

```powershell
python src\infer.py
```

Run inference on specific slices:

```powershell
python src\infer.py 5002 6002 12000
```

Open the interactive viewer:

```powershell
python src\view.py
```

## Training Notes

- training uses `MSELoss` against normalized MIDI velocity targets
- evaluation now uses a contiguous holdout split instead of a random slice split
- a configurable gap between train and test regions reduces near-boundary leakage
- the default training configuration is the notebook-style baseline:
  - `BATCH_SIZE = 16`
  - `NUM_EPOCHS = 150`
  - `LEARNING_RATE = 1e-4`
  - `DROPOUT = 0.5`
- new best-model saves are timestamped and written to `models/`
- training saves both the best test-loss checkpoint and the best test-F1 checkpoint
- resumable state is written to `models/checkpoint_resume.pth`
- `files/training_loss.png` and `files/training_f1.png` are overwritten by each completed training run

## Inference Notes

- `src/infer.py` and `src/view.py` load the checkpoint from `MODEL_LOAD_PATH` in `config.py`
- the baseline model emits raw values, so inference clamps outputs to the normalized target range before hit extraction and plotting
- non-maximum suppression is available in inference to reduce near-duplicate same-note hits

## Validation Status

This workspace has been sanity-checked in `drum310` for:

- config import
- baseline model instantiation
- checkpoint load compatibility
- dataset build
- one forward pass
- one inference slice through the current code path

## Scope

This workspace is intended to be the clean, publishable baseline version built
around the original model behavior. It is not meant to preserve every later
experimental architecture change from the earlier refactor.
