"""Functions to clean and filter SMILES during data preprocessing."""
from rdkit import Chem

from .data_classes import MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS, SUPPORTED_ELEMENTS

def filter_smiles(smi,
                  min_heavy_atoms=MIN_HEAVY_ATOMS,
                  max_heavy_atoms=MAX_HEAVY_ATOMS,
                  supported_elements=SUPPORTED_ELEMENTS,
                  ):
    """
    Filter SMILES based on number of heavy atoms and atom types.

    Args:
        smi (str): SMILES string.
        min_heavy_atoms (int): minimum number of heavy atoms.
        max_heavy_atoms (int): maximum number of heavy atoms.
        supported_elements (set): set of supported elements.

    Returns:
        boolean: whether the SMILES passed the filters.
    """
    mol = Chem.MolFromSmiles(smi)
    
    # reject molecule that is too small or too big
    if not min_heavy_atoms <= mol.GetNumHeavyAtoms() <= max_heavy_atoms:
        return False

    # reject molecule that contains unsupported atom types
    if not all([atom.GetSymbol() in supported_elements for atom in mol.GetAtoms()]):
        return False

    return True


def remove_stereochemistry(smi, canonicalize=True):
    """Remove stereochemistry and optionally canonicalize the SMILES"""
    mol = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=canonicalize)


def clean_smiles(df,
                 min_heavy_atoms=MIN_HEAVY_ATOMS,
                 max_heavy_atoms=MAX_HEAVY_ATOMS,
                 supported_elements=SUPPORTED_ELEMENTS,
                 ):
    """
    Cleans the SMILES from a dataframe:
        - remove any SMILES that cannot be rendered by RDKit
        - apply additional filters based on number of heavy atoms and atom type
        - remove stereochemistry and create canonical SMILES
        - calculate InChI keys and remove duplcate entries
    """
    print('Cleaning dataset...')
    print(f'Dataframe has {len(df)} rows')

    # remove any rows that are missing a SMILES string
    num_nan = len(df[df.SMILES.isna()])
    print(f"Removing {num_nan} rows that are missing a SMILES string")
    df = df[~df.SMILES.isna()]

    # remove any SMILES that cannot be rendered by RDKit
    df['invalid_smiles'] = df.SMILES.apply(lambda smi: True if Chem.MolFromSmiles(smi) is None else False)
    df_invalid = df.query('invalid_smiles == True')
    print(f"Removing {len(df_invalid)} invalid SMILES that could not be rendered by RDKit")
    if len(df_invalid):
        for smi in df_invalid.SMILES:
            print(smi)
    df = df.query('invalid_smiles == False').drop('invalid_smiles', axis=1)   # remove the temporary column

    # apply additional filters based on number of heavy atoms and atom type
    kwargs = {
        'min_heavy_atoms': min_heavy_atoms,
        'max_heavy_atoms': max_heavy_atoms,
        'supported_elements': supported_elements,
    }
    df['filtered_smiles'] = df.SMILES.apply(filter_smiles, **kwargs)
    df_invalid = df.query('filtered_smiles == False')
    print(f"\nRemoving {len(df_invalid)} SMILES that did not pass the filters based on number of heavy atoms and supported atom types")
    if len(df_invalid):
        for smi in df_invalid.SMILES:
            print(smi)
    df = df.query('filtered_smiles == True').drop('filtered_smiles', axis=1)   # remove the temporary column

    # remove stereochemistry and create canonical SMILES
    df.SMILES = df.SMILES.apply(remove_stereochemistry)

    # get InChI keys and remove duplicate compounds
    df['inchi_key'] = df.SMILES.apply(lambda smi: Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smi)))
    num_duplicated = df.duplicated(subset=['inchi_key']).sum()
    print('\nRemoving stereochemistry from all SMILES')
    print(f'Only unique InChI keys will be kept i.e., removing {num_duplicated} compounds that appear multiple times')
    df = df.drop_duplicates(subset='inchi_key')
    print(f'Final cleaned file has {len(df)} rows')
    
    return df
