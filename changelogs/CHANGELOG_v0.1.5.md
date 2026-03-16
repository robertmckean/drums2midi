# CHANGELOG v0.1.5

Finalize the repository structure and documentation for public GitHub review.

## Summary

This release cleans the public repo presentation, adds architecture and summary
documentation, curates example artifacts, and separates local-only tooling from
the public repository view.

## Changes

### Documentation
- Updated `README.md` with corrected local-data wording, example references, and
  architecture doc links
- Added `docs/project_summary_cv_aligned.md`
- Added `docs/system_architecture.md`

### Public Repo Presentation
- Moved the original working notebook to `notebooks/`
- Added curated example artifacts under `examples/`
- Removed internal planning docs from the public docs tree
- Excluded local-only agent instructions and tooling scripts from the public repo

## Notes

- Training data and checkpoints still exist locally and are referenced through
  `config.py`, but they are not included in Git
- The current dataset remains limited and primarily represents a known drum kit,
  so broader generalization would require more diverse data
