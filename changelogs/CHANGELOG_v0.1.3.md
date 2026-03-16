# CHANGELOG v0.1.3

Finalize the viewer/data-loading path so result inspection no longer rewrites
the external processed MIDI CSV during dataset builds.

## Summary

This release keeps the current best checkpoint as the default model load target
and makes `view.py` safe to run without regenerating the processed MIDI CSV in
the external source folder.

## Changes

### Viewer and data loading
- Added an optional `save_processed_csv` flag to `src.data.get_list_of_midi_arrays()`
- Updated `src.robert.process_midi_file()` to skip CSV export when no output path is provided
- Updated `src.view.py` to build labels with `save_processed_csv=False`

### Config
- Set `MODEL_LOAD_PATH` to `models/drum_classifier_20260315_223300_best.pth`

## Validation

- Verified dataset construction in `drum310` with processed CSV writes disabled
- Verified the current best checkpoint loads and the dataset builds to `13680` slices

## Notes

- This avoids sandbox and path issues when viewing results during or after training
- The best-loss checkpoint remains the default load target based on the 250-slice comparison
