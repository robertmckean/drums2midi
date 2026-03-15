# CHANGELOG v0.1.0

Phase 1 extraction of `DrumMidi_Working_LOCAL_00_GM.ipynb` into a
maintainable Python codebase.

## Source

Working notebook: `DrumMidi_Working_LOCAL_00_GM.ipynb`
Original location: `C:\Users\windo\OneDrive\Python\drums_to_midi\00_GM\`

## What was done

### Notebook to src/ extraction

The notebook was split into six Python modules, each responsible for a
single area of functionality:

| File | Purpose | Notebook cells |
|------|---------|----------------|
| `config.py` | All hyperparameters and paths in one place | Scattered constants throughout notebook |
| `src/robert.py` | MIDI CSV parsing, slicing into 2.5s indexed loops | Cell 10 (`process_midi_file`) |
| `src/data.py` | `AudioSliceDataset`, MIDI array to heatmap conversion | Cells 11-12 |
| `src/model.py` | `DrumClassifier` CNN (STFT + encoder-decoder) | Cell 13 |
| `src/train.py` | Training loop with checkpointing and validation | Cells 14-16 |
| `src/infer.py` | CLI inference, heatmap plots, hit extraction | Cells 17-19 |

### Changes from notebook originals

- **librosa replaced with soundfile + scipy**: `librosa.load()` replaced by
  `soundfile.read()` + `scipy.signal.resample_poly()` for audio loading.
  librosa is no longer a dependency.
- **Hardcoded values moved to config.py**: All magic numbers (sample rate,
  slice length, STFT params, paths, training hyperparameters) now live in
  `config.py` and are referenced as `config.PARAM_NAME`.
- **Inline comments instead of docstrings**: All functions documented with
  inline comments per project convention.

### Interactive slice viewer

`src/view.py` added as a lightweight tool for browsing inference results:

- Loads model and dataset once on startup
- Prompts for slice index, pops up a matplotlib GUI window with
  side-by-side heatmaps (model output vs ground truth)
- Prints predicted and ground truth hits to console
- `src/infer.py` modified to move `matplotlib.use("Agg")` into `main()`
  so the GUI backend remains available when view.py imports its functions

### Supporting files

| File | Purpose |
|------|---------|
| `requirements.txt` | pip dependencies (torch, nnAudio, soundfile, scipy, numpy, matplotlib, mido) |
| `.gitignore` | Excludes models/, files/, __pycache__/, .ipynb_checkpoints/ |
| `claude_start.bat` | Windows launch script for Claude Code sessions |
| `CLAUDE.md` | Project instructions and coding rules |

## Verification

- Inference tested on slices 9145 and 100, output matches notebook results
- Interactive viewer tested across multiple slices with GUI heatmap display
- Training script not yet re-run (Phase 2)

## Environment

- Python 3.10, Anaconda env `drum310`
- PyTorch 2.3.1, nnAudio 0.3.1, soundfile, scipy, numpy, matplotlib
- Windows 11 + WSL2, NVIDIA RTX 3080
