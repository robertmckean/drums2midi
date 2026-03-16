# robert.py
# Purpose: MIDI CSV processing — converts one long MIDI file into indexed 2.5s loop slices
# Features: Loop indexing, gap insertion for alignment, time scaling, truncation to 832 steps
# Usage: from src.robert import process_midi_file (called by data.py)

import numpy as np


# Reads raw MIDI CSV (exported from Reaper), slices into 2.5s indexed loops,
# scales note/time columns, inserts gap loops for alignment, truncates to
# new_loop_time steps. Returns (list_of_Nx3_arrays, num_slices).
def process_midi_file(input_file_path, output_file_path, width, note_space, sec, new_loop_time):

    midi_notes = np.genfromtxt(input_file_path, delimiter=',', dtype=int)
    midi_notes = midi_notes.astype(int)

    # Calculate how many MIDI ticks fit in one 2.5s slice (960 ticks/beat at 120 BPM = 4800)
    bpm = 120
    slice_ticks = sec * 960 * bpm / 60
    print('ticks per slice: ', slice_ticks)

    # Add two columns: col[3] = loop_num (which slice this note belongs to),
    # col[4] = offset (tick position within that slice)
    midi_notes = np.concatenate((midi_notes, np.zeros((midi_notes.shape[0], 2), dtype=np.int32)), axis=1)

    # Assign each note to a slice and compute its offset within that slice
    for i, row in enumerate(midi_notes):
        slice_index = int(row[1]) // slice_ticks
        midi_notes[i, 3] = int(row[0]) // slice_ticks
        midi_notes[i, 4] = int((row[0]) - (midi_notes[i, 3] * slice_ticks))

        # Total number of slices is determined by the last note's slice index
        num_slices = midi_notes[-1, 3] + 1

    # Insert blank gap loops where consecutive notes skip a slice index.
    # Without this, silent gaps would corrupt the audio-to-MIDI alignment.
    # Uses a while loop so inserted rows are revisited, filling multi-slice gaps fully.
    i = 0
    while i < len(midi_notes) - 1:
        if midi_notes[i+1][3] - midi_notes[i][3] > 1:
            new_row = midi_notes[i].copy()
            new_row[1] = 0
            new_row[2] = 0
            new_row[3] += 1
            new_row[4] = 0
            midi_notes = np.insert(midi_notes, i+1, new_row, axis=0)
        i += 1

    # Rearrange columns to: note, loop_time, vel, loop_num, raw_time
    midi_notes = midi_notes.take([1, 4, 2, 3, 0], axis=1)

    # Scale note numbers by note_space multiplier and
    # scale loop_time from raw ticks to spectrogram time steps (0 to width-1)
    for i, row in enumerate(midi_notes):
        # loop_time is rescaled into spectrogram-frame coordinates so each MIDI
        # event lines up with the heatmap width used during training/inference.
        midi_notes[i, 1] = np.round((row[1]) * (width-1) / slice_ticks)
        midi_notes[i, 0] = int(row[0]) * note_space

    # Save the fully processed array as CSV for inspection/debugging when requested.
    if output_file_path:
        np.savetxt(output_file_path, midi_notes, delimiter=',', fmt='%d')

    # Creates a list of empty Nx5 arrays, one per slice
    def create_empty_slice_list(num_slices):
        return [np.empty((0, 5), int) for _ in range(num_slices)]

    # Distributes rows into their respective slice arrays based on loop_num (col[3])
    def split_into_slices(midi_notes, midi_np):
        for row in midi_notes:
            midi_np[row[3]] = np.vstack((midi_np[row[3]], row))
        return midi_np

    # Removes notes whose loop_time exceeds the truncation boundary
    def truncate_loop_time(midi_array_list, new_loop_time):
        truncated_midi_array_list = []
        for arr in midi_array_list:
            new_midi_array_list = arr[arr[:, 1] < new_loop_time]
            truncated_midi_array_list.append(new_midi_array_list)
        return truncated_midi_array_list

    # Build empty slice containers, fill them, then strip loop_num and raw_time columns
    midi_np_empty = create_empty_slice_list(num_slices)
    midi_np_slices = split_into_slices(midi_notes, midi_np_empty)

    # Keep only (note, loop_time, vel) — drop loop_num and raw_time columns
    midi_np_slices_3 = [np.delete(arr, [3, 4], axis=1) for arr in midi_np_slices]

    # Truncate notes beyond new_loop_time to match STFT output width (832 steps)
    midi_np_truncated = truncate_loop_time(midi_np_slices_3, new_loop_time)

    # The returned arrays keep only the three columns the downstream heatmap
    # builder needs: note, time index within slice, and velocity.
    return midi_np_truncated, num_slices
