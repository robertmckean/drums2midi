# model.py
# Purpose: Baseline notebook CNN model for drum audio to MIDI conversion
# Features: Trainable STFT, 3-layer encoder, 3-layer decoder, notebook-compatible forward pass
# Usage: from src.model import DrumClassifier

import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import Spectrogram

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# Baseline encoder-decoder CNN that converts raw audio waveform to a (127, 832) MIDI heatmap.
# This matches the original notebook architecture and forward pass.
class DrumClassifier(nn.Module):
    def __init__(self):
        super(DrumClassifier, self).__init__()

        self.spec_layer = Spectrogram.STFT(
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            trainable=True,
            window="hann",
            output_format="Magnitude",
        )

        self.conv1 = nn.Conv1d(in_channels=config.STFT_BINS, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Retained for compatibility with the notebook-era model definition.
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.dropout3 = nn.Dropout(config.DROPOUT)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.deconv1 = nn.ConvTranspose1d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(512, config.N_MIDI_NOTES, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.spec_layer(x)
        # Crop the nnAudio output to the notebook-era dimensions expected by
        # both the encoder/decoder path and the target MIDI heatmaps.
        x = x[:, :config.STFT_BINS, :config.N_TIME_STEPS]

        # Three downsampling conv stages compress time resolution before the
        # decoder reconstructs the full note-by-time heatmap width.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
