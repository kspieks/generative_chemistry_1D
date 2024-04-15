from dataclasses import dataclass, field, fields
from typing import Dict, List

MIN_HEAVY_ATOMS = 10
MAX_HEAVY_ATOMS = 70
SUPPORTED_ELEMENTS = {
    'H',
    'B', 'C', 'N', 'O', 'F',
    'P', 'S', 'Cl',
    'Br', 'I',
}

# supported predictive models with default hyperparameters
RF_HYPERPARAMETERS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42
}
SUPPORTED_PRED_MODELS = {
    'random_forest': RF_HYPERPARAMETERS
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

@dataclass
class PredictiveModel:
    """
    Class to store arguments for training predictive models.

    Args:
        input_file: path to a csv file containing cleaned SMILES and regression targets.
        smi_col: column in input_file containing the SMILES strings.
        regression_targets: list of targets for training the predictive model(s).
                            Must be columns in the csv file from input_file.
        model: string indicating which model type to use.
        hyperparameters: dictionary of hyperparameters to define the model architecture and training.
    """
    input_file: str
    regression_targets: List[str]
    save_dir: str
    model: str
    hyperparameters: Dict = field(default_factory=lambda: dict)

    def __post_init__(self):
        supported_models = list(SUPPORTED_PRED_MODELS.keys())
        if self.model not in supported_models:
            msg = f"Model must be one of the currently supported predictive model types: {supported_models}." 
            msg += f"Got: {self.model}."
            raise ValueError(msg)
        
        # assign default hyperparameters based on model type
        if not self.hyperparameters:
            self.hyperparameters = SUPPORTED_PRED_MODELS[self.model]
