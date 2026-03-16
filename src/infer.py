# infer.py
# Purpose: Inference, visualization, and MIDI hit extraction from trained model
# Features: Heatmap comparison plots, threshold-based hit detection, CLI slice selection
# Usage: python src/infer.py [slice_index ...]

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.data import get_list_of_midi_arrays, process_list_of_midi_arrays, AudioSliceDataset
from src.model import DrumClassifier


# The baseline model is trained against normalized targets but emits unconstrained raw values.
def normalize_output(heatmap):
    return np.clip(heatmap, 0.0, 1.0)


# Runs a single forward pass on one audio slice.
# Returns the model output as a (127, 832) numpy array.
def run_inference(model, dataset, index, device):
    model.eval()
    x_slice = dataset[index][0].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x_slice)
    return output.squeeze(0).cpu().numpy()


# Saves side-by-side heatmaps (model output vs ground truth) as a PNG to files/.
def plot_comparison(output, ground_truth, index):
    os.makedirs(config.FILES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    normalized_output = normalize_output(output)

    axes[0].imshow(normalized_output, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, config.N_TIME_STEPS, 0, config.N_MIDI_NOTES])
    axes[0].set_xlabel("Time (steps)")
    axes[0].set_ylabel("MIDI Note")
    axes[0].set_title(f"Model Output - Slice {index}")

    axes[1].imshow(ground_truth, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, config.N_TIME_STEPS, 0, config.N_MIDI_NOTES])
    axes[1].set_xlabel("Time (steps)")
    axes[1].set_ylabel("MIDI Note")
    axes[1].set_title(f"Ground Truth - Slice {index}")

    save_path = os.path.join(config.FILES_DIR, f"inference_slice_{index}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")


# Extracts predicted hits from model output using velocity threshold,
# then applies non-maximum suppression per note to remove phantom pre-echo hits.
# Model output is clipped to the normalized target range before thresholding.
def extract_midi_hits(heatmap, threshold=config.VELOCITY_THRESHOLD, nms_window=3):
    # Thresholding is done in MIDI velocity units so printed events line up with
    # the project’s human-readable note/time/velocity convention.
    scaled = normalize_output(heatmap) * 127
    indices = np.argwhere(scaled > threshold)
    sorted_indices = indices[indices[:, 1].argsort()]
    hits = []
    for idx in sorted_indices:
        note, time_step = idx
        vel = int(scaled[note, time_step])
        hits.append((note, time_step, vel))

    if nms_window <= 0:
        return hits

    hits_by_note = {}
    for note, time_step, vel in hits:
        hits_by_note.setdefault(note, []).append((time_step, vel))

    suppressed = []
    for note, note_hits in hits_by_note.items():
        note_hits.sort()
        current_group = [note_hits[0]]
        groups = []
        for i in range(1, len(note_hits)):
            # Group nearby same-note activations and keep only the strongest hit
            # in each cluster to reduce duplicates caused by local smearing.
            if note_hits[i][0] - current_group[-1][0] <= nms_window:
                current_group.append(note_hits[i])
            else:
                groups.append(current_group)
                current_group = [note_hits[i]]
        groups.append(current_group)

        for group in groups:
            best = max(group, key=lambda x: x[1])
            suppressed.append((note, best[0], best[1]))

    suppressed.sort(key=lambda x: x[1])
    return suppressed


# Extracts non-zero entries from ground truth heatmap.
def extract_ground_truth_hits(heatmap):
    hits = []
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i][j] != 0:
                hits.append((i, j, heatmap[i][j]))
    hits.sort(key=lambda x: x[1])
    return hits


# Prints predicted and ground truth hits in a formatted table.
def print_hits(predicted_hits, ground_truth_hits):
    print(f"\n{'Predicted Hits':^40} | {'Ground Truth Hits':^40}")
    print("-" * 83)
    print(f"{'note':>6} {'time':>6} {'vel':>6}{'':>22} | {'note':>6} {'time':>6} {'vel':>6}")
    print("-" * 83)

    for note, time_step, vel in predicted_hits:
        print(f"{note:>6} {time_step:>6} {vel:>6}")

    print("-" * 83)
    print("Ground Truth:")
    for note, time_step, vel in ground_truth_hits:
        print(f"{note:>6} {time_step:>6} {int(vel):>6}")

    print(f"\nPredicted: {len(predicted_hits)} hits, Ground Truth: {len(ground_truth_hits)} hits")


# CLI entry point: loads model, builds dataset, runs inference on requested slice indices.
def main():
    matplotlib.use("Agg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if len(sys.argv) > 1:
        indices = [int(arg) for arg in sys.argv[1:]]
    else:
        indices = [config.INFERENCE_INDEX]

    model = DrumClassifier().to(device)
    model.load_state_dict(torch.load(config.MODEL_LOAD_PATH, map_location=device))
    print(f"Loaded model from: {config.MODEL_LOAD_PATH}")

    # Rebuild the aligned dataset from source files so slice indices match the
    # current config and the checkpoint’s expected target layout.
    list_of_midi_arrays, num_slices = get_list_of_midi_arrays()
    list_of_processed_midi = process_list_of_midi_arrays(list_of_midi_arrays)

    dataset = AudioSliceDataset(
        config.get_audio_file(),
        list_of_processed_midi,
        slice_len_sec=config.SLICE_LEN_SEC,
    )
    print(f"Dataset size: {len(dataset)}")

    for index in indices:
        if index >= len(dataset):
            print(f"\nSlice {index} out of range (dataset has {len(dataset)} slices), skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running inference on slice {index}")
        print(f"{'=' * 60}")

        output = run_inference(model, dataset, index, device)
        ground_truth = list_of_processed_midi[index]

        plot_comparison(output, ground_truth, index)

        predicted_hits = extract_midi_hits(output)
        ground_truth_hits = extract_ground_truth_hits(ground_truth)
        print_hits(predicted_hits, ground_truth_hits)


if __name__ == "__main__":
    main()
