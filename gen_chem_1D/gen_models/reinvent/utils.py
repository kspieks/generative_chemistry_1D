"""Collection of helper functions related to generative modeling."""
import numpy as np
import pandas as pd
from rdkit import Chem
import torch


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


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
                inchikeys.append(Chem.inchi.MolToInchiKey(mol))
        except:
            pass
    return valid_smiles, inchikeys


def get_valid_unique_smiles_idx(smiles):
    """Takes a list of SMILES and returns list and index of the valid and unique InChI keys."""
    inchi_keys = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            inchi_keys.append(Chem.inchi.MolToInchiKey(mol))
        else:
            inchi_keys.append('')
    
    df = pd.DataFrame({'smiles': smiles, 'inchi_keys': inchi_keys})
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
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


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
