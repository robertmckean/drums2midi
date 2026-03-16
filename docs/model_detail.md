# Model Detail

## Executive Summary

- The baseline model is a compact end-to-end audio-to-MIDI architecture with a
  trainable STFT-based spectral front end and a symmetric 3-stage
  encoder/decoder.
- It contains `6,952,959` trainable parameters.
- About `4.20M` parameters sit in the trainable STFT front end, with the
  remainder in the convolutional encoder/decoder path.
- The model is designed to take a fixed `2.5` second audio slice and emit a
  dense MIDI-style heatmap representing note activity over time.
- The architecture is intentionally controlled and notebook-compatible rather
  than optimized for maximum scale or broad generalization.

## Current Baseline Configuration

From `config.py` and `src/model.py`:

- Sample rate: `44,100 Hz`
- Slice length: `2.5 s`
- Input waveform length: `110,250` samples
- Input waveform format: 'mono'
- STFT:
  - `n_fft = 2048`
  - `win_length = 2048`
  - `hop_length = 128`
  - raw STFT output: `(1025, 862)`
  - cropped STFT input to CNN: `(1024, 832)`
- Output heatmap:
  - MIDI notes: `127`
  - time steps: `832`

## Architecture Summary

The current baseline model is:

1. Trainable STFT front end (using nnAudio)
2. Three 1D convolution + pooling encoder stages
3. Three 1D transposed-convolution decoder stages
4. Final decoder output producing a dense MIDI activity heatmap

## Forward Pass Overview

```text
Waveform
-> STFT front end
-> convolutional encoder
-> compressed latent time representation
-> transposed-convolution decoder
-> MIDI activity heatmap
```

The forward path is:

```text
Waveform (110250 samples)
-> trainable STFT
-> crop to (1024, 832)
-> Conv1d 1024 -> 512 + MaxPool
-> Conv1d 512 -> 256 + MaxPool
-> Conv1d 256 -> 128 + MaxPool
-> ConvTranspose1d 128 -> 256
-> ConvTranspose1d 256 -> 512
-> ConvTranspose1d 512 -> 127
-> Output heatmap (127, 832)
```

## Tensor Shape Progression

Measured from a live forward-pass probe:

| Stage | Shape |
| --- | --- |
| Input waveform | `(1, 110250)` |
| Raw STFT magnitude | `(1, 1025, 862)` |
| Cropped STFT tensor | `(1, 1024, 832)` |
| Encoder stage 1 output | `(1, 512, 416)` |
| Encoder stage 2 output | `(1, 256, 208)` |
| Encoder stage 3 output | `(1, 128, 104)` |
| Decoder stage 1 output | `(1, 256, 208)` |
| Decoder stage 2 output | `(1, 512, 416)` |
| Final output | `(1, 127, 832)` |

## Parameter Count

Exact model parameter count from the current implementation:

- Total parameters: `6,952,959`
- Trainable parameters: `6,952,959`

### Per-Layer Breakdown

| Layer | Type | Parameters |
| --- | --- | ---: |
| `spec_layer` | Trainable STFT | `4,198,400` |
| `conv1` | Conv1d `1024 -> 512` | `1,573,376` |
| `conv2` | Conv1d `512 -> 256` | `393,472` |
| `conv3` | Conv1d `256 -> 128` | `98,432` |
| `bn1` | BatchNorm1d `512` | `1,024` |
| `bn2` | BatchNorm1d `256` | `512` |
| `bn3` | BatchNorm1d `128` | `256` |
| `deconv1` | ConvTranspose1d `128 -> 256` | `98,560` |
| `deconv2` | ConvTranspose1d `256 -> 512` | `393,728` |
| `deconv3` | ConvTranspose1d `512 -> 127` | `195,199` |

## Architectural Notes

- The trainable STFT front end is the single largest parameter block in the
  model, accounting for roughly 60% of all parameters.
- The encoder reduces temporal resolution from `832` steps to `104` steps
  through three pooling stages, then the decoder reconstructs the full time
  axis.
- The output head emits raw values rather than sigmoid-bounded probabilities;
  inference clamps output values before hit extraction.
- BatchNorm and Dropout layers are defined in the model class but are not
  currently used in the forward path, reflecting earlier notebook
  experimentation.

## What We Can State Confidently

These are directly supported by the current code and runtime probe:

- exact total parameter count
- exact per-layer parameter counts
- exact tensor shapes through the forward path
- exact STFT, slice, and output dimensions
- exact encoder/decoder depth and channel progression

## What Should Be Framed More Carefully

These should be described as design characteristics, not inflated claims:

- The model is a controlled baseline, not a large-scale production architecture
  optimized for broad generalization.
- The current dataset is limited and primarily represents several known drum kits, 
  so stronger generalization would require broader training data.
- The architecture demonstrates end-to-end AI system design discipline, but it
  should not be presented as state-of-the-art or broadly benchmarked.
