# 1D Generative Modeling for Chemistry

Generates SMILES strings with a generative model and biases the generator towards better
molecular properties using predictive models.

- **Generative model**: REINVENT, an RNN that generates SMILES token by token. A prior is
  pre-trained on a set of SMILES, and an agent initialized from the prior is then fine-tuned with
  rewards. Any token-based SMILES generator could be added.
- **Predictive models**: random forest regressors, used to score the generated molecules and as
  rewards when biasing the agent.
- **Rewards**: a predicted property or an RDKit property (e.g., molecular weight) converted to a
  score with a worst, acceptable, and best value or an acceptable window, plus penalties for
  substructures given as SMILES or SMARTS.
- **Scaffold-constrained generation**: sample only around the `(*)` attachment points of a scaffold.
- **Postprocessing**: neutralize, remove duplicates and training molecules, enumerate
  stereoisomers, cluster with UMAP and k-means, and find the most similar training molecules.

Todo: add support for more expressive predictive models (e.g., Chemprop)

## Main commands

```bash
# once, after cloning
brew install uv
uv sync

# run the fully automated example (about 3 minutes on a MacBook CPU)
cd examples/fully_automated
uv run bash run_fully_automated.sh

# or run the step-by-step example one script at a time
cd examples/step_by_step
uv run bash 1_preprocess_pred_data.sh

# open the notebooks (or `uv run jupyter notebook`)
uv run jupyter lab
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`). Python 3.12; minimum versions are
declared in `pyproject.toml` and the exact versions used are locked in `uv.lock`.

```bash
uv sync    # creates .venv and installs the package in editable mode
```

`uv sync` also installs the packages needed for the notebooks (JupyterLab, ipykernel, matplotlib,
seaborn). To skip them, use `uv sync --no-default-groups`.

To run `python` without the `uv run` prefix, activate the environment once per terminal:

```bash
source .venv/bin/activate
```

PyTorch comes from PyPI. The models run on CUDA if available and otherwise on the CPU. To force
a device, set `GEN_CHEM_1D_DEVICE` to `cpu`, `cuda`, or `mps` (Apple GPU):

```bash
GEN_CHEM_1D_DEVICE=mps uv run bash run_fully_automated.sh
```

MPS is never chosen automatically. It trains the prior 2.5x faster per epoch (2.5 s vs 6.4 s) and
runs the fully automated example 1.2x faster (145 s vs 178 s), but scaffold-constrained sampling is
6x slower (546 s vs 90 s). MPS also lacks float64, and PyTorch may lack some MPS
operators (`PYTORCH_ENABLE_MPS_FALLBACK=1` runs those on the CPU).

The models run in float32, PyTorch's default, and the reward follows the network dtype. In a test, a
float64 network took about 3x longer (fully automated example on the CPU: 794 s vs 257 s) and showed
no detectable difference in the generated molecules.

## Workflow

Every step reads a yaml file (`input.yml`) and runs through a script in `gen_chem_1D/`.

| Step | Script | Does |
|---|---|---|
| Preprocess predictive data | `pred_models/preprocess_pred_data.py` | filter by size and elements, canonicalize |
| Train predictive models | `pred_models/train_pred_models.py` | random forest, 80:20 evaluation, refit on all data |
| Preprocess generative data | `gen_models/reinvent/preprocess_gen_data.py` | same filters, writes `.smi` and `vocab.txt` |
| Train the prior | `gen_models/reinvent/train_prior.py` | pre-train the RNN |
| Sample | `gen_models/reinvent/sample.py` | sample, check validity, remove duplicates, score |
| Bias the prior | `gen_models/reinvent/train_agent.py` | fine-tune an agent with rewards |
| Postprocess | `data/postprocess/postprocess_generated_data.py` | see below |

The two examples show the whole workflow:

| Folder | Dataset | Scripts |
|---|---|---|
| `examples/fully_automated` | ESOL solubility, 1,128 molecules | `run_fully_automated.sh` runs every step |
| `examples/step_by_step` | Lipophilicity (logD), 4,200 molecules | numbered scripts `1_...` to `9_...`, see its README |

Run each script from its own folder with `uv run bash <script>.sh`.

Training and sampling are not seeded, so reruns generate different molecules. Compare
summary statistics (validity, uniqueness, property distributions) instead of individual SMILES.

`pretrained_models/reinvent/chembl` holds a prior pre-trained on ChEMBL. See its README.

### Postprocessing

The script uses relative imports, so run it as a module and not by file path:

```bash
uv run python -m gen_chem_1D.data.postprocess.postprocess_generated_data --yaml_file input.yml
```

The yaml file needs a `postprocess_data` section with `input_files` and `training_data`.
Other options and their defaults are in the `Postprocess` dataclass in
`gen_chem_1D/data/data_classes.py`. The similarity step writes
`similarity_matrix_<N>x<M>.pkl` to the current folder.

## Notebooks

```bash
uv run jupyter lab        # file browser, tabs, and terminals in one window
uv run jupyter notebook   # one notebook per browser tab
```

Both open the same `.ipynb` files and use the project environment's kernel. Start them with
`uv run` (or after `source .venv/bin/activate`) so the command comes from `.venv`. A bare
`jupyter notebook` outside the environment can start another Python's Jupyter, such as
Anaconda's, which does not have rdkit.

In VS Code, select `.venv/bin/python` as the notebook kernel instead.

| Notebook | Contents |
|---|---|
| `examples/fully_automated/0_visualize_esol_data.ipynb` | ESOL logSolubility, heavy atoms, and token counts |
| `examples/fully_automated/1_analyze_prior.ipynb` | Molecules sampled from the prior |
| `examples/fully_automated/2_analyze_agent.ipynb` | Molecules sampled from the biased agent |
| `examples/step_by_step/0_visualize_lipo_data.ipynb` | Lipophilicity data |
| `examples/step_by_step/5_analyze_prior.ipynb` | Prior samples |
| `examples/step_by_step/7_analyze_agent1.ipynb` | Agent after one-sided logD bias |
| `examples/step_by_step/9_analyze_agent2.ipynb` | Agent with logD sweet spot, MW, and substructure rewards |
| `notebooks/scaffold_constrained/sample_scaffold_constrained.ipynb` | Scaffold-constrained sampling for fexofenadine |

## Updating package versions

```bash
uv lock --upgrade   # newest versions allowed by pyproject.toml
uv sync
```

Rerun the examples and notebooks to check the outputs, then commit the updated `uv.lock`.

## Project structure

```
├── gen_chem_1D/
│   ├── data/                   # dataclasses for the yaml settings, cleaning, postprocessing
│   ├── gen_models/reinvent/    # REINVENT: model, tokenization, train prior, sample, bias
│   ├── pred_models/            # random forest, features, scoring functions
│   └── utils/                  # yaml parsing
├── examples/                   # fully automated and step-by-step workflows
├── notebooks/                  # scaffold-constrained generation
├── pretrained_models/          # ChEMBL prior
├── pyproject.toml
└── uv.lock
```
