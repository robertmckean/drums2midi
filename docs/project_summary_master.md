# Master Project Summary — drums2midi

drums2midi is a machine learning system that converts recorded drum audio into structured MIDI events.

The project includes the full pipeline required to prepare data, train the model, evaluate results, and generate predictions from new audio:

- generation, alignment, and curation of audio–MIDI training data
- feature extraction using a trainable Short Time Fourier Transform (STFT) front end
- a custom convolutional encoder–decoder model implemented in PyTorch
- tools for training, checkpointing, prediction, and interactive inspection of model output

The repository is a refactored Python implementation of a working research notebook. It preserves the training workflow, model architecture, and evaluation approach used to develop the system while making the code easier to review, run, and maintain.

Training data and model checkpoints are stored locally and are not included in the repository. The current dataset represents a limited number of drum kit configurations, so the trained model performs best on kits similar to those used during training. Broader generalization would require a more diverse training corpus rather than major architectural changes.

## What This Project Demonstrates

This project demonstrates the ability to design and implement an end-to-end machine learning pipeline, including:

- generation and preparation of supervised audio–MIDI datasets
- integration of signal processing and deep learning
- disciplined training and checkpoint management
- evaluation of model behavior using controlled validation data
- conversion of experimental research code into a maintainable codebase

The project serves as a working baseline for audio-to-MIDI transcription using an end-to-end learning approach.
