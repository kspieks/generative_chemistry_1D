from dataclasses import dataclass, field, fields

MIN_HEAVY_ATOMS = 10
MAX_HEAVY_ATOMS = 70
SUPPORTED_ELEMENTS = {
    'H',
    'B', 'C', 'N', 'O', 'F',
    'P', 'S', 'Cl',
    'Br', 'I',
}

SUPPORTED_PRED_MODELS = {
    'random_forest'
}

@dataclass
class Preprocess:
    """
    Class to store settings for cleaning SMILES during data pre-processing.

    Args:
        pred_input_file: path to a csv file with SMILES to be cleaned before training a predictive model.
        pred_output_file: path to write the cleaned csv file for training a predictive model.

        gen_input_file: path to a csv file with SMILES to be cleaned before training a generative model.
        gen_output_file: path to write the cleaned SMILES for training a generative model.

        min_heavy_atoms: minimum number of heavy atoms.
        max_heavy_atoms: maximum number of heavy atoms.
        supported_elements: set of supported atomic symbols.
    """
    pred_input_file: str = 'data.csv'
    pred_output_file: str = 'cleaned_data.csv'

    gen_input_file: str = 'smiles.csv'
    gen_output_file: str = 'cleaned_mols.smi'
    gen_vocab_file: str = 'vocab.txt'

    min_heavy_atoms: int = MIN_HEAVY_ATOMS
    max_heavy_atoms: int = MAX_HEAVY_ATOMS
    supported_elements: set = field(default_factory=lambda: SUPPORTED_ELEMENTS)

    def __post_init__(self):
        for field in fields(self):
            setattr(self, field.name, field.type(getattr(self, field.name)))
