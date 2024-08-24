#!/usr/bin/env python
import os

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from gen_chem_1D.pred_models.features.create_features import create_features
from gen_chem_1D.pred_models.rf_model import predict_rf


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

    def convert_values_to_scores(self, values):
        # already in range 0, 1
        if len(self.scale_params) == 1:
            return values
    
        # acceptable value, worst, best
        elif len(self.scale_params) == 3:
            return (values - self.w) / (self.b - self.w)
        
        # acceptable low, acceptable high, min value, max value
        elif len(self.scale_params) == 4:
            return 1.0 - np.abs(self.b - values) / self.w
        
    def __call__(self, smiles_list):
        """Make predictions for a given list of SMILES"""
        values = self.predict(smiles_list)

        return self.convert_values_to_scores(values)

    def predict(self, smiles_list):
        pass


class CalcProp(ConvertScore):
    """Scores structures based on calculated properties from RDKit."""
    def __init__(self, name, scale):
        super().__init__(scale_params=scale)
        
        name = name.lower()
        if name == 'mw':
            self.func = Descriptors.MolWt
        elif name == 'tpsa':
            self.func = Descriptors.TPSA
        elif name == 'logp':
            self.func = Descriptors.MolLogP
        elif name == 'hbd':
            self.func = Descriptors.NumHDonors
        elif name == 'hba':
            self.func = Descriptors.NumHAcceptors
        elif name == 'rotb':
            self.func = Descriptors.NumRotatableBonds
        elif name == 'fracsp3':
            self.func = Descriptors.FractionCSP3
        elif name == 'coo_counts':
            self.func = Chem.Fragments.fr_COO
        elif name == 'num_aromatic_rings':
            def is_ring_aromatic(mol, bond_ring):
                """
                Helper function for identifying if a bond is aromatic.

                Args:
                    mol: RDKit molecule.
                    bond_ring: RDKit bond from a ring.

                Returns:
                    True if the bond is in an aromatic ring. False otherwise.
                """
                for idx in bond_ring:
                    if not mol.GetBondWithIdx(idx).GetIsAromatic():
                        return False
                return True

            def get_num_aromatic_rings(mol):
                num_aromatic_rings = 0
                for ring_bonds in mol.GetRingInfo().BondRings():
                    num_aromatic_rings += is_ring_aromatic(mol, ring_bonds)
                return num_aromatic_rings
            
            self.func = get_num_aromatic_rings

    def predict(self, smiles_list):
        out = np.zeros(len(smiles_list))
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            out[i] = self.func(mol)

        return out


class RFPredictor(ConvertScore):
    def __init__(self, model_path, scale=[0.5, 0., 1.]):
        super().__init__(scale_params=scale)
        self.model_path = model_path
    
    def predict(self, smiles_list):
        out = predict_rf(smiles_list, self.model_path)
        return out


class MultiScore():
    """Class that combines multiple scorers"""
    def __init__(self, scoring_functions):
        scorers=[]
        names = []
        for sf, params in scoring_functions.items():
            if sf.lower() in ['mw', 'tpsa', 'logp', 'hbd', 'hba', 'rotb', 'fracsp3', 'coo_counts', 'num_aromatic_rings']:
                scorers.append(CalcProp(name=sf, scale=params))
            elif sf.split('_')[0] == 'rf':
                scorers.append(RFPredictor(**params))
            names.append(sf)

        self.scorers = scorers
        self.names = names
        self.acclist = [sc.acc for sc in self.scorers]

    def __call__(self, smiles_list):
        """Get total score for each SMILES and fraction of SMILES above acceptable threshold"""
        scores = np.vstack([s(smiles_list) for s in self.scorers]).T

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
        scores = (scores - mint) / (maxt - mint)
        scores = np.clip(scores, 0, 1)

        # scalarize scores for each compound
        scalar_scores = np.sum(scores * weights, axis=1)

        # rescale scores st the min value is 0, max value is 1
        scalar_scores = (scalar_scores - np.min(scalar_scores)) / (np.max(scalar_scores) - np.min(scalar_scores))

        #return np.sum(scores*weights, axis=1), frac
        return scalar_scores, frac


class Scorer():
    def __init__(self, scoring_functions=None):
        self.scoring_functions = MultiScore(scoring_functions)
        self.names = self.scoring_functions.names

    def __call__(self, smiles_list):
        """Get total weighted score for each smiles"""
        scores, frac = self.scoring_functions(smiles_list)
        return scores, frac
