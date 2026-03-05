# Copilot / AI Agent Instructions — ParamEst-NN

Short, practical guidance to make an AI coding agent immediately productive in this repository.

1) Big picture
- Purpose: code and notebooks implement neural-network-based parameter estimation from photon-counting trajectories (see paper referenced in README).
- Major components:
  - `notebooks/` : exploratory notebooks that generate trajectories, train models, and produce results. Key notebooks: `1-Trajectories_generation.ipynb`, `2-Training.ipynb`, `3-Results.ipynb`.
  - `paramest_nn/` : the installable Python package with core code — domain logic lives here (`custom_layers.py`, `quantum_tools.py`).
  - `data/` : expected runtime dataset layout; contains `training-trajectories`, `validation-trajectories`, and `models` subfolders used by notebooks and scripts.
  - `scripts/` : small CLI scripts for batch post-processing (e.g. `uniform_2d.py`, `concat_results.py`).

2) How data flows and runtime expectations
- `notebooks/*` set a `datapath` variable at runtime; code reads/writes under that location. Agents should not hardcode absolute paths — follow that variable.
- Training produces models to `data/models/*` and expects validation trajectories in `data/validation-trajectories/*` when reproducing results.
- The 2D posterior pipeline uses `scripts/uniform_2d.py` which is intended to be run in parallel across cluster nodes; its input is an integer index selecting a ground-truth parameter pair.

3) Developer workflows & commands (explicit)
- Create environment (recommended conda):
  - `conda create -n paramest python=3.9` then `conda activate paramest`
- Install package for local edits:
  - `pip install -e .` (run from repo root)
- Run notebook kernels locally: use Jupyter Lab/Notebook and ensure `datapath` points to the `data/` folder (or downloaded dataset).
- Download dataset (README documents Zenodo link). Some notebooks can auto-download when `download_required = True` but require `wget`.
- Run 2D Ultranest script usage:
  - `python scripts/uniform_2d.py --help`
  - After producing CSV results, merge with: `python scripts/concat_results.py`

4) Project-specific conventions & patterns
- Notebooks are first-class — many workflows are implemented as notebooks rather than CLI entrypoints. When converting or automating tasks prefer reusing functions in `paramest_nn` rather than executing notebook cells.
- Package code is the authoritative place for reusable logic: prefer editing `paramest_nn/*.py` over changing notebook cell logic.
- TensorFlow version noted in README: tested with TF 2.12.1. GPU/TPU differences: `2-Training.ipynb` contains TPU-specific setup for Colab — be cautious when running locally.
- Data and model filenames follow directory conventions under `data/` (see `data/training-trajectories/`, `data/models/`); scripts rely on these exact layouts.

5) Integration points & external deps
- Core ML framework: TensorFlow (notebooks mention TF 2.12.1). Models saved in Keras/HDF5/`.keras` formats.
- Ultranest is used for Bayesian posterior computations; scripts call it (see `scripts/uniform_2d.py`).
- The code expects NumPy `.npy` datasets for trajectories and CSV outputs for Ultranest post-processing.

6) Useful files to inspect when making changes
- [notebooks/2-Training.ipynb] - training workflow and Colab/TPU notes
- [paramest_nn/custom_layers.py] - custom Keras layers used by models
- [paramest_nn/quantum_tools.py] - domain-specific data generation and helpers
- [scripts/uniform_2d.py] - batch posterior computation (Ultranest)
- [scripts/concat_results.py] - how CSV outputs are merged for plotting

7) Typical small tasks and hints for agents
- To add a new training experiment: implement reusable functions in `paramest_nn/` and call them from a new notebook or a small CLI script under `scripts/`.
- To reproduce a figure from the paper: follow notebook `3-Results.ipynb` and ensure `data/models` and `data/validation-trajectories` are populated (see README instructions and `scripts/concat_results.py`).
- For Windows users: prefer conda environment; `pip install -e .` still required to import `paramest_nn` in notebooks.

8) What NOT to change without confirmation
- Do not alter the data layout under `data/` or rename expected `.npy` files — notebooks and scripts assume those exact names.
- Avoid making large API changes to `paramest_nn` without updating notebooks that import it.

9) Where to ask follow-ups
- Primary contact: repository README (author GitHub/email).

If anything here is unclear or you want more examples (e.g., snippet edits in `paramest_nn/custom_layers.py` or a runnable CLI wrapper for notebook flows), tell me which area to expand.
