# view.py
# Purpose: Interactive slice viewer for browsing inference results with pop-up heatmaps
# Features: Loads model once, loops on user input, shows matplotlib GUI windows
# Usage: python src/view.py

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import torch
import matplotlib.pyplot as plt

from src.data import get_list_of_midi_arrays, process_list_of_midi_arrays, AudioSliceDataset
from src.model import DrumClassifier
from src.infer import run_inference, extract_midi_hits, extract_ground_truth_hits, print_hits, normalize_output


# Displays side-by-side heatmaps in an interactive matplotlib window.
# Unlike plot_comparison() in infer.py, this uses plt.show() for GUI display
# instead of plt.savefig(). Window is blocking — close it to return to the prompt.
def show_comparison(output, ground_truth, index):
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))

    # Left panel: model prediction
    axes[0].imshow(normalize_output(output), origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, config.N_TIME_STEPS, 0, config.N_MIDI_NOTES])
    axes[0].set_xlabel("Time (steps)")
    axes[0].set_ylabel("MIDI Note")
    axes[0].set_title(f"Model Output — Slice {index}")

    # Right panel: ground truth from MIDI labels
    axes[1].imshow(ground_truth, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, config.N_TIME_STEPS, 0, config.N_MIDI_NOTES])
    axes[1].set_xlabel("Time (steps)")
    axes[1].set_ylabel("MIDI Note")
    axes[1].set_title(f"Ground Truth — Slice {index}")

    plt.tight_layout()
    plt.show()


# Interactive entry point: loads model and dataset once, then loops on user input.
# Each slice number pops up a heatmap window and prints hits to console.
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained model weights
    model = DrumClassifier().to(device)
    model.load_state_dict(torch.load(config.MODEL_LOAD_PATH, map_location=device))
    print(f"Loaded model from: {config.MODEL_LOAD_PATH}")

    # Build dataset (same pipeline as training — needed for both X audio and Y labels)
    list_of_midi_arrays, num_slices = get_list_of_midi_arrays()
    list_of_processed_midi = process_list_of_midi_arrays(list_of_midi_arrays)

    dataset = AudioSliceDataset(
        config.get_audio_file(),
        list_of_processed_midi,
        slice_len_sec=config.SLICE_LEN_SEC,
    )

    max_index = len(dataset) - 1
    print(f"Dataset size: {len(dataset)} slices (valid range: 0-{max_index})")

    # Interactive loop — prompts for slice index until user quits
    while True:
        try:
            user_input = input(f"\nEnter slice index (0-{max_index}, q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("q", "quit"):
            break

        # Validate input is a number within range
        try:
            index = int(user_input)
        except ValueError:
            print(f"Invalid input: '{user_input}' — enter a number or 'q'")
            continue

        if index < 0 or index > max_index:
            print(f"Index {index} out of range (0-{max_index})")
            continue

        # Run inference, print hits first so they're visible while viewing the heatmap
        print(f"Running inference on slice {index}...")
        output = run_inference(model, dataset, index, device)
        ground_truth = list_of_processed_midi[index]

        predicted_hits = extract_midi_hits(output)
        ground_truth_hits = extract_ground_truth_hits(ground_truth)
        print_hits(predicted_hits, ground_truth_hits)

        show_comparison(output, ground_truth, index)


if __name__ == "__main__":
    main()

