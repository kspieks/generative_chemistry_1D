# 1D Generative Modeling for Chemistry
The goal of this repo is to utilize generative modeling techniques to create SMILES strings of molecules. Property prediction models can be used to push the generative models into a better property space. Currently, the repo supports REINVENT as the generative model, but any token-based SMILES generator could also be implemented.

## Pip installation instructions
As of April 2024, the [PyTorch](https://pytorch.org/get-started/locally/) website has the following statements:
- "PyTorch is supported on macOS 10.15 (Catalina) or above."
- "It is recommended that you use Python 3.8 - 3.11"

```
# create conda env
conda create -n gen_chem_1d python=3.11.8 -y

# activate conda env
conda activate gen_chem_1d

# install PyTorch for CPU only
pip install torch torchvision torchaudio

## install PyTorch for CUDA 11.8
## pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install rdkit
pip install rdkit

# install other packages
pip install joblib jupyter scikit-learn seaborn tqdm

# install repo in editable mode
pip install -e .
```
