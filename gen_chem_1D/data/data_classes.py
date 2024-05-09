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
        save_dir: directory to save predictive model(s) to.
    """
    input_file: str
    regression_targets: List[str]
    model: str
    hyperparameters: Dict = field(default_factory=lambda: dict)
    save_dir: str = 'pred_model'

    def __post_init__(self):
        supported_models = list(SUPPORTED_PRED_MODELS.keys())
        if self.model not in supported_models:
            msg = f"Model must be one of the currently supported predictive model types: {supported_models}." 
            msg += f"Got: {self.model}."
            raise ValueError(msg)
        
        # assign default hyperparameters based on model type
        if not self.hyperparameters:
            self.hyperparameters = SUPPORTED_PRED_MODELS[self.model]

@dataclass
class GenerativePrior:
    """
    Class to store arguments for training a generative prior.
    
    Args:
        vocab_file: text file containing tokens that define the vocabulary.
        smi_path: path to a file containing cleaned SMILES.
        checkpoint_path: optional path to a pre-trained generative model.
        out_dir: directory to write models to.
        num_epochs: number of training epochs.
        batch_size: number of sequences to sample at each iteration.
        init_lr: initial learning rate.
    """
    vocab_file: str
    smi_path: str
    out_dir: str
    checkpoint_path: str = ''

    num_epochs: int = 20
    batch_size: int = 128
    init_lr: float = 1e-3

    def __post_init__(self):
        for field in fields(self):
            setattr(self, field.name, field.type(getattr(self, field.name)))

@dataclass
class GenerativeBias:
    """
    Class to store arguments for biasing a generative prior.
    
    Args:
        vocab_file: text file containing tokens that define the vocabulary.
        prior_checkpoint_path: path to an RNN checkpoint file to use as a Prior.
        agent_checkpoint_path: path to an RNN checkpoint file to use as a Agent. Defaults to Prior checkpoint.
        agent_save_path: path to save the biased agent to.
        num_steps: number of training interations.
        batch_size: number of sequences to sample at each iteration.
        init_lr: initial learning rate.
        reward_multiplier: factor used in calculating augmented log-likelihood.
    """
    vocab_file: str
    prior_checkpoint_path: str
    agent_checkpoint_path: str = ''
    agent_save_path: str = 'Agent.ckpt'
    
    num_steps: int = 20
    batch_size: int = 64
    init_lr: float = 5e-4
    reward_multiplier: float = 80.0

    scoring_functions: Dict = field(default_factory=lambda: dict())
    substructs: Dict = field(default_factory=lambda: dict)

    def __post_init__(self):
        self.init_lr = float(self.init_lr)
        # by default, restore Agent to the same model as Prior
        if not self.agent_checkpoint_path:
            self.agent_checkpoint_path = self.prior_checkpoint_path

@dataclass
class GenerativeSample:
    """
    Class to store arguments for sampling a generative model.

    Args:
        checkpoint_path: path to a trained generative model.
        vocab_file: text file containing tokens that define the vocabulary.
        num_smiles: number of SMILES to generate.
        max_len: maximum sequence length that will be generated.
        output_file: csv filepath to write results to.
        batch_size: number of sequences to sample at each iteration.
        scaffold_constraint: optional SMILES containing * for controlled generation e.g., CC(*)CC.
    """
    checkpoint_path: str
    vocab_file: str
    num_smiles: int
    max_len: int
    output_file: str

    pred_models: Dict
    calculators: List[str] = field(default_factory=lambda: ['mw'])
    
    batch_size: int = 128
    scaffold_constraint: str = ''
