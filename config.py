# config.py
# Purpose: Central configuration for all project hyperparameters and paths
# Features: Audio, MIDI, model, and training settings in one place
# Usage: Import config and reference config.PARAM_NAME throughout the project

import os
from datetime import datetime

# Paths
BASE_DIR = r"C:\Users\windo\VS_Code\drums2midi"
SOURCE_DIR = r"C:\Users\windo\OneDrive\Python\drums_to_midi\00_GM"

MIDI_DIR = os.path.join(SOURCE_DIR, "midi")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FILES_DIR = os.path.join(BASE_DIR, "files")

AUDIO_FILE = os.path.join(MIDI_DIR, "reaper_GM_audio_1.wav")
FALLBACK_AUDIO_FILE = os.path.join(MIDI_DIR, "reaper_GM_audio.wav")
MIDI_CSV_INPUT = os.path.join(MIDI_DIR, "reaper_midi_GM.csv")
MIDI_CSV_OUTPUT = os.path.join(MIDI_DIR, "reaper_GM_midi_processed.csv")

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "drum_classifier_best.pth")
MODEL_LOAD_PATH = os.path.join(
    MODELS_DIR,
    "drum_classifier_20260315_210518_best.pth",
)
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "checkpoint_resume.pth")

# Audio
SAMPLE_RATE = 44100
SLICE_LEN_SEC = 2.5
SLICE_LEN_SAMPLES = int(SLICE_LEN_SEC * SAMPLE_RATE)

# MIDI processing
BPM = 120
TICKS_PER_BEAT = 960
SLICE_TICKS = SLICE_LEN_SEC * TICKS_PER_BEAT * BPM / 60
NOTE_SPACE = 1
STFT_WIDTH = 862
NEW_LOOP_TIME = 832

# Model
N_FFT = 2048
HOP_LENGTH = 128
WIN_LENGTH = 2048
STFT_BINS = 1024
N_MIDI_NOTES = 127
N_TIME_STEPS = 832
DROPOUT = 0.5

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 150
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8
TRAIN_TEST_GAP_MINUTES = 10
RANDOM_SEED = 42
CLIP_VALUE = 0.5
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 10

# Inference
VELOCITY_THRESHOLD = 9.5
HIT_THRESHOLD = VELOCITY_THRESHOLD / 127
INFERENCE_INDEX = 9145


def get_audio_file():
    if os.path.exists(AUDIO_FILE):
        return AUDIO_FILE
    return FALLBACK_AUDIO_FILE


# Returns a timestamped path for saving a new trained model (never overwrites existing models)
def get_model_save_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(MODELS_DIR, f"drum_classifier_{timestamp}_best.pth")
