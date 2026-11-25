"""Collection of helper functions related to generative modeling."""
import functools
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
import torch


@functools.lru_cache(maxsize=None)
def get_device():
    """
    Returns the torch device for the REINVENT models: CUDA if available, otherwise the CPU.

    Set the environment variable GEN_CHEM_1D_DEVICE to cpu, cuda, or mps to force a device,
    e.g., GEN_CHEM_1D_DEVICE=mps uv run bash run_fully_automated.sh. An unavailable device
    raises an error. The result is cached, so set the variable before the first call.

    MPS (Apple GPU) is never chosen automatically. Compared with the CPU and CUDA:
    - Scaffold-constrained sampling is much slower on MPS than on the CPU (546 s vs 90 s)
      because it currently loops over single molecules.
    - float64 is not supported, so tensors must be float32 (the reward follows the network dtype).
    - PyTorch may lack some MPS operators; PYTORCH_ENABLE_MPS_FALLBACK=1 runs those on the CPU.
    - The PyTorch 2.15 docs list macOS 15.0 or newer.
    - Results differ slightly from CPU and CUDA, so runs are not reproducible across devices.
    """
    name = os.environ.get('GEN_CHEM_1D_DEVICE', 'auto').lower()
    available = {'cpu': True,
                 'cuda': torch.cuda.is_available(),
                 'mps': torch.backends.mps.is_available(),
                 }
    if name == 'auto':
        name = 'cuda' if available['cuda'] else 'cpu'
    elif name not in available:
        raise ValueError(f"GEN_CHEM_1D_DEVICE must be auto, cpu, cuda, or mps, got '{name}'")
    elif not available[name]:
        raise RuntimeError(f"GEN_CHEM_1D_DEVICE={name} but {name} is not available")
    return torch.device(name)


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the device from get_device(). Be aware in case some
       operations are better left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    return torch.autograd.Variable(tensor).to(get_device())


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def seq_to_smiles(seqs, voc):
    """
    Takes an output sequence from the RNN and returns the corresponding SMILES.
    
    Args:
        seqs: generated sequences of token indices.
        voc: Vocabulary instance.
    
    Returns:
        list of SMILES strings created by decoding the sequence of indices.
    """
    smiles_list = []
    for seq in seqs.cpu().numpy():
        smiles_list.append(voc.decode(seq))
    return smiles_list


def validate_smiles(smiles_list):
    """
    Takes a list of generated SMILES and returns the valid SMILES with corresponding InChI key.
    Currently not used since the function below is more comprehensive.
    """
    valid_smiles = []
    inchikeys = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_smiles.append(Chem.MolToSmiles(mol))
                inchikeys.append(inchi.MolToInchiKey(mol))
        except:
            pass
    return valid_smiles, inchikeys


def get_valid_unique_smiles_idx(smiles):
    """Takes a list of SMILES and returns list and index of the valid and unique InChI keys."""
    canonical_smiles = []
    inchi_keys = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            inchi_keys.append(inchi.MolToInchiKey(mol))
        else:
            canonical_smiles.append('')
            inchi_keys.append('')
    
    df = pd.DataFrame({'smiles': canonical_smiles, 'inchi_keys': inchi_keys})
    df_valid = df[df.inchi_keys!='']
    v_smiles = df_valid.smiles.values.tolist()
    v_inchi_key = df_valid.inchi_keys.values.tolist()

    df_valid_unique = df[(~df.inchi_keys.duplicated()) & (df.inchi_keys!='')]
    vu_smiles = df_valid_unique.smiles.values.tolist()
    vu_inchi_key = df_valid_unique.inchi_keys.values.tolist()
    vu_indices = list(df_valid_unique.index)
    ndup = len(df[(df.inchi_keys.duplicated()) & (df.inchi_keys!='')])

    return v_smiles, v_inchi_key, vu_smiles, vu_inchi_key, vu_indices, ndup


def get_unique_indices(arr):
    # find unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    return torch.LongTensor(np.sort(idxs)).to(get_device())


def get_ss_score(smiles, patts):
    """
    Finds if compounds from list of smiles contain substructure from patts.

    Args:
        smiles: list of generated SMILES strings.
        patts: list of RDKit mols to use for substructure matching.
    
    Returns:
        matrix of size num smiles x num patterns. 1 indicates match. 0 otherwise.
    """
    match = np.zeros((len(smiles), len(patts)))
    for si, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol:
            for pi, p in enumerate(patts):
                if mol.GetSubstructMatch(p): match[si, pi] = 1
    return match
