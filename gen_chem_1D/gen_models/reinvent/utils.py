"""Collection of helper functions related to generative modeling."""
import numpy as np
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
    """Takes a list of generated SMILES and returns the valid SMILES with corresponding InChI key."""
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
