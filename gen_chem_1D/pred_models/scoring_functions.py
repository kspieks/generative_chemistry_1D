#!/usr/bin/env python
import pickle as pkl

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from .features.featurizers import calc_atompair_fp, calc_donorpair_fp

class ConvertScore():
    """Class for converting a predicted or calculated value to a score bounded by 0 to 1"""
    def __init__(self, scale_params=[]):
        self.scale_params = scale_params

        # already in range 0, 1
        if len(self.scale_params) == 1:
            self.acc = self.scale_params[0]

        # acceptable value, worst, best
        elif len(self.scale_params) == 3:
            self.b = self.scale_params[2]
            self.w = self.scale_params[1]
            self.acc = (self.scale_params[0] - self.w)/ (self.b - self.w)
        
        # acceptable low, acceptable high, min value, max value
        elif len(self.scale_params) == 4:
            self.b = (self.scale_params[0] + self.scale_params[1]) / 2
            self.w = max(abs(self.scale_params[2] - self.b), abs(self.scale_params[3] - self.b))
            self.acc = 1.0 - abs(self.b - self.scale_params[0]) / self.w

    def convert_to_score(self, value):
        # already in range 0, 1
        if len(self.scale_params) == 1:
            return value  
    
        # acceptable value, worst, best
        elif len(self.scale_params) == 3:
            return (value - self.w) / (self.b - self.w)
        
        # acceptable low, acceptable high, min value, max value
        elif len(self.scale_params) == 4:
            return 1.0 - np.abs(self.b - value) / self.w
        
    def __call__(self, smiles):
        out = self.predict(smiles)
        if out:
            return self.convert_to_score(out)
        else:
            return 0.0

    def predict(self, smiles):
        pass


class CalcProp(ConvertScore):
    """Scores structures based on calculated properties."""
    def __init__(self, name, scale):
        super().__init__(scale_params=scale)
        
        name = name.lower()
        if name == 'mw':
            self.func = Descriptors.MolWt
        elif name == 'logp':
            self.func = Descriptors.MolLogP
        elif name == 'hbd':
            self.func = Descriptors.NumHDonors
        elif name == 'hba':
            self.func = Descriptors.NumHAcceptors
        elif name == 'rotb':
            self.func = Descriptors.NumRotatableBonds
        elif name == 'coo_counts':
            self.func = Chem.Fragments.fr_COO

    def predict(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        out = 0.0
        if mol:
            try: out = self.func(mol)
            except: out = 0.0
        return out


class RFPredictor(ConvertScore):
    def __init__(self, model_path, scale=[0.5, 0., 1.]):
        super().__init__(scale_params=scale)

        # self.model_path = model_path
        with open(model_path, 'rb') as f:
            self.rf = pkl.load(f)
    
    def predict(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        out = None
        if mol:
            try:
                ap_fp = calc_atompair_fp(smiles)
                dp_fp = calc_donorpair_fp(smiles)

                X = np.hstack([ap_fp, dp_fp])
                # reshape to num samples x feature size
                out = self.rf.predict(X.reshape(1, -1))[0]
            except:
                out = None
        return out


class MultiScore():
    """Class that combines multiple scorers"""
    def __init__(self, scoring_functions):
        scorers=[]
        names = []
        for sf, params in scoring_functions.items():
            if sf.lower() in ['mw', 'logp', 'hbd', 'hba', 'rotb', 'coo_counts']:
                scorers.append(CalcProp(name=sf, **params))
            elif sf.split('_')[0] == 'rf':
                scorers.append(RFPredictor(**params))
            names.append(sf)

        self.scorers = scorers
        self.names = names
        self.acclist = [sc.acc for sc in self.scorers]

    # get total score for each smiles and fraction of smiles above acceptable threshold
    def __call__(self, smiles):
        scores = np.array([[s(smi) for s in self.scorers] for smi in smiles])

        # clip: negative values are set to 0; >1 set to 1
        scores = np.clip(scores, 0, 1)

        # frac of smiles above acceptable threshold
        frac = np.sum(scores > self.acclist, axis=0) / len(scores)

        # weighted by inverse of frac
        frac2 = np.array([f if f>0 else 0.001 for f in frac])
        weights = 1 / frac2 / np.sum(1 / frac2)

        # norm each objective individually, st 0,1 are at 1 std from mean
        std_tmp = np.std(scores, axis=0)
        mean_tmp = np.mean(scores, axis=0)
        mint = mean_tmp - std_tmp
        maxt = mean_tmp + std_tmp
        scores = (scores-mint) / (maxt-mint)
        scores = np.clip(scores, 0, 1)

        # scalarize scores for each compound
        scalar_scores = np.sum(scores * weights, axis=1)

        # rescale scores st the min value is 0, max value is 1
        scalar_scores = (scalar_scores - np.min(scalar_scores)) / (np.max(scalar_scores) - np.min(scalar_scores))

        #return np.sum(scores*weights, axis=1), frac
        return scalar_scores, frac

    def score(self, smiles):
        return [s(smiles) for s in self.scorers]

    # get individual predictions
    def predict(self, smiles):
        return [s.predict(smiles) for s in self.scorers]


class Scorer():
    def __init__(self, scoring_functions=None):
        self.scoring_functions = MultiScore(scoring_functions)
        self.names = self.scoring_functions.names

    # get total weighted score for each smiles
    def __call__(self, smiles):
        scores, frac = self.scoring_functions(smiles)
        return scores, frac

    # get individual scores for each scorer, without any reweighting
    def score(self, smiles):
        scores = [self.scoring_functions.score(smi) for smi in smiles]
        return np.array(scores, dtype=np.float)

    # get individual predictions for each predictor
    def predict(self, smiles):
        preds = [self.scoring_functions.predict(smi) for smi in smiles]
        return np.array(preds, dtype=np.float)
