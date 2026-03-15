# data.py
# Purpose: Audio and MIDI data loading, slicing, and DataLoader construction
# Features: AudioSliceDataset, MIDI array processing, train/test split
# Usage: from src.data import AudioSliceDataset, build_dataloaders

import os
import sys
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Add project root to path so config is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import robert


# Wraps robert.process_midi_file() with config values.
# Returns (list_of_Nx3_arrays, num_slices) where each array has columns: note, loop_time, vel.
def get_list_of_midi_arrays():
    list_of_midi_arrays, num_slices = robert.process_midi_file(
        config.MIDI_CSV_INPUT,
        config.MIDI_CSV_OUTPUT,
        width=config.STFT_WIDTH,
        note_space=config.NOTE_SPACE,
        sec=config.SLICE_LEN_SEC,
        new_loop_time=config.NEW_LOOP_TIME,
    )
    return list_of_midi_arrays, num_slices


# Converts each Nx3 sparse array (note, time, vel) into a (127, 832) dense heatmap.
# Each cell [note, time_step] = velocity value. Zero means no hit at that position.
def process_list_of_midi_arrays(list_of_midi_arrays):
    list_of_processed_midi = []

    for idx, midi_array in enumerate(list_of_midi_arrays):
        # Initialize empty heatmap for this slice
        processed_midi = np.zeros((config.N_MIDI_NOTES, config.N_TIME_STEPS))

        # Place each hit into the heatmap at [note, time] = velocity
        for row in midi_array:
            drum_id = int(row[0])
            time_slice_idx = int(row[1])
            velocity = row[2]
            processed_midi[drum_id, time_slice_idx] = velocity

        list_of_processed_midi.append(processed_midi)

    return list_of_processed_midi


# Dataset that pairs 2.5s audio slices with their corresponding (127, 832) MIDI heatmaps.
# Audio is loaded once in __init__ and split into fixed-length chunks.
class AudioSliceDataset(Dataset):
    def __init__(self, audio_path, midi_data, slice_len_sec, sr=config.SAMPLE_RATE):
        # Load full WAV file as float32 mono (no resampling — source is already 44100 Hz)
        self.audio, _ = sf.read(audio_path, dtype="float32")
        self.slice_len = int(slice_len_sec * sr)
        self.slices = self._slice_audio()
        self.midi_data = midi_data

    # Splits the full audio waveform into non-overlapping fixed-length chunks.
    # Any trailing samples shorter than slice_len are discarded.
    def _slice_audio(self):
        num_slices = len(self.audio) // self.slice_len
        return [self.audio[i * self.slice_len:(i + 1) * self.slice_len] for i in range(num_slices)]

    def __len__(self):
        return len(self.slices)

    # Returns (audio_tensor, midi_tensor) pair for one slice.
    # MIDI velocities are normalized from 0-127 to 0-1 range for MSE loss.
    def __getitem__(self, idx):
        audio_slice = torch.from_numpy(self.slices[idx]).float()
        midi_slice = torch.from_numpy(self.midi_data[idx]).float()
        midi_slice /= 127
        return audio_slice, midi_slice


# Builds contiguous train/test subsets with a time gap between them.
def _build_contiguous_split(dataset):
    dataset_len = len(dataset)
    gap_slices = int((config.TRAIN_TEST_GAP_MINUTES * 60) / config.SLICE_LEN_SEC)
    usable_len = dataset_len - gap_slices
    if usable_len < 2:
        raise ValueError(
            "Train/test gap is too large for the available dataset length."
        )

    train_len = int(usable_len * config.TRAIN_SPLIT)
    test_len = usable_len - train_len
    if train_len < 1 or test_len < 1:
        raise ValueError(
            "Contiguous split produced an empty train or test set."
        )

    train_indices = list(range(train_len))
    test_start = train_len + gap_slices
    test_indices = list(range(test_start, test_start + test_len))
    split_info = {
        "gap_slices": gap_slices,
        "train_start": 0,
        "train_end": train_len - 1,
        "test_start": test_start,
        "test_end": test_start + test_len - 1,
    }
    return Subset(dataset, train_indices), Subset(dataset, test_indices), split_info


# Builds train/test DataLoaders from the full pipeline.
# Calls get_list_of_midi_arrays -> process_list_of_midi_arrays -> AudioSliceDataset,
# then creates a contiguous train block and later test block separated by a gap.
def build_dataloaders():
    list_of_midi_arrays, num_slices = get_list_of_midi_arrays()
    list_of_processed_midi = process_list_of_midi_arrays(list_of_midi_arrays)

    dataset = AudioSliceDataset(
        config.get_audio_file(),
        list_of_processed_midi,
        slice_len_sec=config.SLICE_LEN_SEC,
    )

    train_set, test_set, split_info = _build_contiguous_split(dataset)

    loader_generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True, generator=loader_generator
    )
    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False
    )

    print(
        "Contiguous split: "
        f"train slices {split_info['train_start']}-{split_info['train_end']}, "
        f"gap {split_info['gap_slices']} slices "
        f"({config.TRAIN_TEST_GAP_MINUTES} minutes), "
        f"test slices {split_info['test_start']}-{split_info['test_end']}"
    )

    return train_loader, test_loader, dataset

