"""
Functions for featurizing an individual molecule via RDKit.
For each function, the input is a SMILES string, and the return is a 1D numpy array.
"""
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.AtomPairs import Sheridan

from .utils import rdkit_to_np, _hash_fold


# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.GetHashedAtomPairFingerprint
# https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html#rdkit.Chem.rdFingerprintGenerator.GetAtomPairGenerator
def calc_atompair_fp(smi,
                     count=True,
                     minDistance=1,
                     maxDistance=30,
                     fpSize=1024,
                     includeChirality=True,
                     ):
    """
    Publication: Carhart, R.E. et al. "Atom Pairs as Molecular Features in 
    Structure-Activity Studies: Definition and Applications” J. Chem. Inf. Comp. Sci. 25:64-73 (1985).

    "An atom pair substructure is defined as a triplet of two non-hydrogen atoms and their shortest
    path distance in the molecular graph, i.e. (atom type 1, atom type 2, geodesic distance).
    In the standard RDKit implementation, distinct atom types are defined by tuples of atomic number, 
    number of heavy atom neighbours, aromaticity and chirality. All unique triplets in a molecule
    are enumerated and stored in sparse count or bit vector format."
    https://www.blopig.com/blog/2022/06/exploring-topological-fingerprints-in-rdkit/
    """
    mol = Chem.MolFromSmiles(smi)
    atompair_gen = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=minDistance,
        maxDistance=maxDistance,
        fpSize=fpSize,
        includeChirality=includeChirality,
    )
    fp = getattr(atompair_gen,
                 f'Get{"Count" if count else ""}Fingerprint'
                 )(mol)

    return rdkit_to_np(fp, fpSize)


# https://rdkit.org/docs/source/rdkit.Chem.AtomPairs.Sheridan.html
def calc_donorpair_fp(smi, fpSize=1024):
    mol = Chem.MolFromSmiles(smi)
    sparse_vec = Sheridan.GetBPFingerprint(mol)
    nze = sparse_vec.GetNonzeroElements()

    return _hash_fold(nze, fpSize)
