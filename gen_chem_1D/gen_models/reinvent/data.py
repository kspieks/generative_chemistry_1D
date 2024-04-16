import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import Dataset

from .utils import Variable


class MolData(Dataset):
    """
    Custom PyTorch Dataset that takes a file containing SMILES.

    Args:
        file_path: path to a file containing SMILES strings separated by newlines.
        voc: a Vocabulary instance

    Returns:
        Custom PyTorch dataset for training the RNN model.
    """
    def __init__(self, file_path, voc):
        self.voc = voc
        self.smiles = []
        with open(file_path, 'r') as f:
            for line in f:
                self.smiles.append(line.strip())

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized_mol = self.voc.tokenize(mol)
        encoded_mol = self.voc.encode(tokenized_mol)
        return Variable(encoded_mol)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return f"Dataset containing {len(self)} molecules."

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


class Experience(object):
    """
    Class for prioritized experience replay that remembers the highest scored sequences
    seen and samples from them with probabilities relative to their scores.
    """
    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key = lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            # print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError(f'Size of memory ({len(self)}) is less than requested sample ({n})')
        else:
            scores = np.array([x[1] for x in self.memory])
            sample = np.random.choice(len(self), size=n, replace=False, p=scores / np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, file_path, scoring_function, Prior):
        """
        Adds experience from a file with SMILES.
        Needs a scoring function and an RNN to score the sequences.
        Using this feature means that the learning can be very biased and is typically advised against.
        """
        with open(file_path, 'r') as f:
            smiles = []
            for line in f:
                smile = line.strip()
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def __len__(self):
        return len(self.memory)
