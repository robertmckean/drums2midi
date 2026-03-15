# Codex Task Brief: Prepare the Drum-to-MIDI Project for Professional Portfolio Publishing

## Objective

Review the existing project documentation and source tree, then plan and execute a **lightweight professionalization pass** so this project can be presented as a credible portfolio artifact for senior technical, AI, and fractional CTO positioning.

This is **not** a request to rebuild the project, reproduce the original training pipeline, or make the full model/dataset reproducible.

The goal is to make a technical reviewer think:

> This person clearly designed and built a real end-to-end ML system.

The result should be a repo/documentation package that is easy to skim in 5–10 minutes and communicates:
* current hands-on AI/ML capability
* system design ability
* engineering judgment
* clarity of thought
* technical credibility

---

## Critical Context

### What this project was
This project was an end-to-end machine learning system for converting **drum audio into structured MIDI events**.

It included work across:
* dataset creation / curation
* audio preprocessing
* STFT-based feature extraction
* model training
* inference / decoding
* MIDI event generation
* evaluation / experimentation

### What is no longer available
The original training dataset has been deleted. It was very large (hundreds of GB including intermediate artifacts / derived data) and will **not** be rebuilt.

That means:
* do **not** plan work that depends on recovering the original dataset
* do **not** plan work that requires retraining from scratch
* do **not** present reproducibility as a project requirement
* do **not** leave TODOs that assume the dataset will later be restored

### Publishing intent
This repository is being prepared as a **professional proof-of-capability artifact**, not as an open-source research package intended for external users to reproduce training.

That means the main audience is:
* CTOs
* technical hiring managers
* senior engineers
* founders
* consulting prospects

Most readers will:
* read the README
* skim the architecture explanation
* look at a few code files
* maybe inspect examples
* evaluate whether the repo feels real, thoughtful, and technically grounded

That is the bar.

---

## High-Level Deliverables

Codex should review the existing repo and documentation, then produce a clean publishable version with the following core outputs:

1. **A strong README.md**
2. **A lightweight architecture/system overview**
3. **A clear explanation of the dataset situation**
4. **A cleaned repo structure where practical**
5. **Removal or isolation of confusing / low-value clutter**
6. **Optional sample/demo assets if already available or easy to create**
7. **A short project summary that can be reused externally**
8. **A plan summary describing what was changed and why**

---

## What Success Looks Like

A reviewer should be able to quickly understand:

* what the system does
* why it is hard
* how it was approached
* what core pipeline components exist
* what the model/pipeline architecture roughly is
* what engineering problems were encountered
* what was learned
* why the dataset is missing without the repo feeling broken

The repo should feel like:
* a serious engineering project
* a credible independent AI effort
* a project owned by someone who understands systems, not just notebooks

The repo should **not** feel like:
* an abandoned experiment
* a broken research repo
* a tutorial copy
* a half-finished toy demo
* a promise of future work

---

## Constraints

### Do NOT do these things
* Do not assume the original dataset can be recovered
* Do not require the user to regenerate the dataset
* Do not design around full reproducibility
* Do not invent benchmark numbers or fake metrics
* Do not overstate results that are not documented in the project
* Do not add marketing fluff or generic AI hype language
* Do not convert this into a giant engineering cleanup project
* Do not propose major refactors unless clearly necessary for presentation
* Do not leave “future work” sections that imply missing critical functionality
* Do not create work that has little visibility payoff

### Preferred posture
Be pragmatic. Favor:
* clarity over completeness
* credibility over polish theater
* strong explanations over heavy rebuild work
* representative code organization over perfection
* professional technical writing over sales language

---

## Codex Process

## Phase 1: Review and Inventory

First, inspect the existing repository and documentation.

Produce an internal assessment of:
* current folder structure
* existing README and supporting docs
* main entry points
* training-related code
* inference-related code
* dataset/preprocessing code
* evaluation code
* any diagrams, notes, examples, or sample assets
* dead files, redundant files, confusing file names, abandoned drafts, scratch files, debug artifacts

Also determine:
* whether there are any easily usable demo inputs/outputs already present
* whether architecture can be described confidently from the codebase
* whether there are any documented results or observations that can be stated honestly

Important: infer as much as possible from the repo itself. Do not require new explanation from the user unless absolutely unavoidable.

---

## Phase 2: Create a Publishing Plan

After review, produce a concise execution plan before making large changes.

The plan should identify:
* what docs need to be written or rewritten
* what code/files should be renamed or reorganized
* what low-value clutter should be removed or hidden
* whether a lightweight architecture document is useful
* whether a sample/demo folder is possible
* which existing files best demonstrate the system

The plan should explicitly respect the deleted-dataset constraint.

---

## Phase 3: Execute

Then implement the plan.

Prioritize the following.

### 1. README.md
This is the most important artifact.

The README should likely include sections close to the following, adapted to the actual repo:

# Project Title
Use a clear professional title such as:
**Machine Learning System for Converting Drum Audio to MIDI**
or another title that better matches the actual implementation.

# Overview
A concise description of the project and what it demonstrates.

# Problem
Explain why drum-audio-to-MIDI is non-trivial:
* transient timing sensitivity
* overlapping percussion events
* class ambiguity
* signal processing challenges
* event reconstruction complexity

# System Approach
Describe the pipeline at a high level, for example:
Audio → preprocessing / segmentation → STFT feature extraction → model inference/training → event decoding → MIDI generation

Only describe what the repo actually supports.

# Repository Structure
Summarize the key folders/files and what they do.

# Dataset
This section is mandatory.

It should clearly explain:
* the original dataset was curated and used for training
* it is not included in the repo
* it will not be rebuilt for this publication pass
* the omission is due to size / practicality / artifact volume, and possibly licensing if applicable
* the repo is being published as a technical portfolio artifact, not a full reproducibility package

This section should sound normal and professional, not apologetic.

Suggested tone:
> The original training dataset is not included in this repository due to its size and the volume of derived intermediate artifacts. This repository is intended to document the system design, pipeline structure, and implementation approach rather than serve as a fully reproducible training package.

Adjust wording as needed to match the facts.

# Pipeline / Architecture
Explain the main system components.
If helpful, include a simple text diagram or Mermaid diagram only if appropriate and stable.

# Model / Learning Approach
Describe the learning architecture honestly based on the code.
Do not embellish.

# Inference / Output
Explain how audio becomes structured MIDI output.

# Results / Observations
Only include what is supported by the repo or existing notes.
If hard numbers are not available, use engineering observations rather than invented metrics.

# Lessons Learned
This section is very important.
Capture engineering judgment such as:
* dataset quality mattered more than expected
* timing precision was sensitive to preprocessing choices
* overlapping events created ambiguous labels
* end-to-end pipeline behavior mattered more than isolated model tuning
* audio representation choices materially affected outcome quality

Use only lessons that can be grounded in actual project reality.

# Status
A short note that this is an archived independent project / portfolio artifact, if appropriate.

---

### 2. Repo Structure Cleanup
If the current repo structure is chaotic, make it easier to skim.

Possible goals:
* group related code by purpose
* isolate old experiments / scratch work
* improve file naming consistency
* reduce visual clutter at top level

Do this conservatively.
Do not break working code unnecessarily.
Do not do a deep architecture refactor just for aesthetics.

If reorganization risk is high, prefer documenting the structure clearly instead of moving everything.

---

### 3. Architecture Overview
Create a lightweight supporting document if helpful, such as:
* `architecture.md`
* `system_overview.md`

This should explain:
* major components
* data flow
* what each stage is responsible for
* where model training fits
* where inference fits
* where MIDI generation fits

This doc should be concise and skimmable.

---

### 4. Representative Samples
If the repo already contains or can easily support examples, create a `samples/` or `examples/` folder.

Useful assets could include:
* a short input audio file
* a resulting MIDI file
* a screenshot
* a small diagram
* a generated visualization

Only do this if it is low effort and grounded in existing project artifacts.
Do not fabricate outputs.
Do not create a major workstream around sample generation.

---

### 5. Remove or Isolate Noise
Look for:
* obsolete temp files
* “final_v2_really_final” style artifacts
* dead notebooks
* duplicate docs
* experimental scripts with confusing names
* logs / caches / generated junk
* files that make the repo look messy without adding value

Either remove, rename, relocate, or clearly mark them.

Be conservative with deletion if unsure; moving to an `archive/` or `experimental/` folder may be safer.

---

### 6. Reusable External Summary
Create a short summary file or section that can be reused later for LinkedIn / CV / portfolio copy.

Something in this spirit:

> Built an end-to-end machine learning pipeline for converting drum audio into structured MIDI events, including dataset engineering, STFT-based feature extraction, model training, and MIDI event reconstruction. The project served as an independent deep-dive into modern ML workflows, signal processing, and end-to-end AI system design.

Adjust to actual implementation details.
This should be technically strong, brief, and credible.

A file like `project_summary.md` is acceptable if useful.

---

## Tone and Writing Requirements

The writing must sound like a serious engineer wrote it.

Use:
* direct language
* calm confidence
* precise statements
* grounded technical descriptions

Avoid:
* exaggerated claims
* startup hype
* “revolutionary”
* “state-of-the-art” unless actually evidenced
* generic AI buzzwords
* empty optimism

This project should read like:
**experienced technical leadership with hands-on implementation credibility**

---

## Code Review Priorities

When reviewing code, prioritize what affects presentation and credibility.

Look for:
* obviously broken imports or entry points
* misleading or unclear file names
* comments that are stale or inaccurate
* dead code that confuses the architecture story
* scripts that are central enough to mention in docs
* whether there is a clear separation between preprocessing, model logic, training, inference, and output generation

Do not turn this into a full rewrite unless the current state makes the repo incomprehensible.

---

## Deliverable Format for the User

At the end of the work, provide:

1. A concise summary of what was changed
2. The final proposed repo structure
3. Any important assumptions made
4. Any files that still need user validation
5. Any risks or limitations that remain
6. Suggested next action for publishing (for example: GitHub repo cleanup complete, ready for README review)

---

## Practical Decision Rules

When choosing between two approaches, prefer the one that:
* reduces user effort
* improves first-impression credibility
* avoids reliance on missing data
* keeps claims honest
* makes the repo easier to understand quickly

If something would take significant effort but add little visible value, skip it.

If something is imperfect but understandable, document it rather than rebuilding it.

If a reviewer would never notice the difference, do not spend time on it.

---

## Summary of the Core Ask

Please review the existing project, determine the most practical path to turn it into a strong professional portfolio artifact, then implement that path with an emphasis on:

* README quality
* architecture clarity
* honest handling of the missing dataset
* repo cleanliness
* technical credibility
* minimal unnecessary work

This is a **presentation and credibility optimization pass**, not a research reproduction effort.
