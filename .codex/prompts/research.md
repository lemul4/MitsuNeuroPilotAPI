You are a senior ML/Software engineer and architect. Your task is to conduct research and write detailed documentation about the CODE AND FUNCTIONALITY that exists ONLY within a specific folder of the repository.

# Project Context (for terminology)

Project: Python end-to-end neural network autopilot with the following pipeline:

- expert/driver and dataset collection in CARLA
- model training
- model testing in CARLA

Folder `<TARGET_DIR>` = `lead/common`

For this task, you must document ONLY the folder: `<TARGET_DIR>`.

# Strict Scope Boundaries

1) Analyze and open files ONLY inside `<TARGET_DIR>`.

2) DO NOT open files outside `<TARGET_DIR>`.

- If you see imports from other folders/files, ONLY INDICATE them (what is imported and from where).
- If the behavior cannot be explained correctly without reading an external file, ask me 1–3 clarification questions and explicitly specify which external file is required and why.

3) DO NOT modify code and DO NOT create files.
No patches, refactoring, or formatting.

4) DO NOT run long computations (training, CARLA simulations, benchmarks).
Only lightweight commands for structural analysis are allowed:

- `ls`
- `find`
- `rg`
- file viewing

5) DO NOT use the network or web search for this task (we are documenting the local code in the folder).
If you believe external documentation is required (for example the CARLA API), ask for permission first.

# Accuracy Requirements

- Do not invent facts about the code. If you are unsure, mark it as **“not found”** or **“unclear”**.
- Do not invent line numbers. If you reference a line, ensure it actually exists (by inspecting the file and numbering).

# What Needs to Be Done (Algorithm)

## 0) Verify the folder

Check that the folder `<TARGET_DIR>` exists and that it is the correct folder.

## 1) Extract the folder structure

Produce the folder structure:

- tree of files/subfolders (depth 3–5)
- mark file types (`py`, `yaml`, `json`, `toml`, `sh`, `md`, etc.)

## 2) Find entry points within the folder

Search for:

- run scripts (`if __name__ == "__main__"`)
- CLI entrypoints
- main modules (`dataset`, `model`, `train`, `eval`, `inference`, `sim`)
- configuration files (`yaml`, `json`, `toml`) if present

## 3) Build an import map for files in `<TARGET_DIR>`

For each file, determine which imports exist:

a) **Inside `<TARGET_DIR>`** (internal imports within the folder)

b) **Outside `<TARGET_DIR>`** (other repository folders — indicate the module/path exactly as written in the import)

c) **External dependencies** (for example: `torch`, `numpy`, `carla`, `hydra`, etc.)

Additionally indicate which **NAMES** (classes/functions) are imported from external modules.

## 4) For each `.py` file in the folder, produce a detailed description

### File purpose

1–3 sentences describing what the file does.

### Main entities

Classes:

- purpose
- key fields
- key methods
- where they are used

Functions:

- purpose
- inputs
- outputs
- side effects

### Execution flow

Explain how execution proceeds:

- which functions are called
- order of calls

### I/O and artifacts

What the code reads/writes:

- files
- directories
- formats

What CLI arguments or configuration files are expected.

Indicate CARLA-related dependencies if present:

- simulator
- sensors
- controllers

### Important parameters / hyperparameters

If present.

### Errors and exceptions

What can fail and where.

### TODO / FIXME

If present.

## 5) Assemble the final document in Markdown research_<TARGET_DIR>.md

# Final Document Structure (strictly follow)

# `<TARGET_DIR>` — Folder Documentation

## 1. Purpose of the Folder

What this folder does within the pipeline (based only on the code inside this folder).

## 2. How to Use It (if applicable)

- Which commands/scripts are run from this folder (do not invent — only if visible in the code or README inside the folder).
- What input data is required.
- What output artifacts are produced.

## 3. Folder Structure

Provide the file tree (as a code block).

## 4. Module and Import Map

### 4.1 Internal dependencies (inside `<TARGET_DIR>`)

Table:

file → what it imports → from where → purpose

### 4.2 Dependencies outside the folder

Table:

file → what it imports → “from where (as written in the import)” → comment

DO NOT open external files. Only record the import.

### 4.3 External libraries

List external libraries and where they are used (based on imports).

## 5. File Details

For each file (ordered by importance):

### `<relative/path/to/file.py>`

- purpose
- all classes/functions (with signatures)
- execution flow (brief but clear)
- inputs/outputs/artifacts
- imports (brief: internal/external)
- nuances / pitfalls

## 6. Data Flow and Artifacts (if applicable)

- What data passes through this folder (datasets/checkpoints/logs/metrics).
- Where it is expected to be located on disk (if visible in the code).

# Style

- Do not include large code blocks. Only short fragments (up to ~20 lines) if absolutely necessary for understanding.
- When referencing code locations use: 'path/to/file.py:line' ONLY if the line number has been verified.
Otherwise reference: 'path/to/file.py' + function/class name

# Start of the Task

1. Verify the folder `<TARGET_DIR>`.
2. Output the folder tree.
3. Build the import map.
4. Describe the files.
5. Create a file named: research_<TARGET_DIR>.md inside the folder `<TARGET_DIR>` containing the full research and documentation.
