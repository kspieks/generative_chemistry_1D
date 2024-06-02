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
        gen_vocab_file: path to write the vocabulary to.

        min_heavy_atoms: minimum number of heavy atoms.
        max_heavy_atoms: maximum number of heavy atoms.
        supported_elements: set of supported atomic symbols.
        remove_stereo: boolean indicating whether to remove stereochemistry.
        canonicalize: boolean indicating whether to canonicalize the SMILES.
    """
    pred_input_file: str = 'data.csv'
    pred_output_file: str = 'cleaned_data.csv'

    gen_input_file: str = 'smiles.csv'
    gen_output_file: str = 'cleaned_mols.smi'
    gen_vocab_file: str = 'vocab.txt'

    min_heavy_atoms: int = MIN_HEAVY_ATOMS
    max_heavy_atoms: int = MAX_HEAVY_ATOMS
    supported_elements: set = field(default_factory=lambda: SUPPORTED_ELEMENTS)
    remove_stereo: bool = True
    canonicalize: bool = True

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
        num_steps: after the specified number of steps, decrease the learning rate and print info about model validation.
        decrease_lr: after the specified number of steps, multiply the learning by (1 - decrease_lr).
        save_limit: number of intermediate models to save.
        embedding_size: dimension of the embedding.
        hidden_size: dimension of the hidden layers.
        dropout_input: dropout applied to the embeddings before input to RNN.
        dropout_hidden: dropout applied between hidden layers of RNN.
        temperature: RNN temperature, alters diversity by rescaling logits.
        max_len: maximum sequence length that will be generated when validating the model.
    """
    vocab_file: str
    smi_path: str
    out_dir: str
    checkpoint_path: str = ''

    num_epochs: int = 20
    batch_size: int = 128
    init_lr: float = 1e-3
    num_steps: int = 50
    decrease_lr: float = 0.03
    save_limit: int = 2

    embedding_size: int = 128
    hidden_size: int = 512
    dropout_input: float = 0
    dropout_hidden: float = 0
    temperature: float = 1.0

    max_len: int = 140

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
        scoring_functions: dictionary with scoring function to use. Example:
            {'rf_logSolubility': 
                {'model_path': 'pred_model/rf_lipo.pkl',
                 'scale': [-1, -1.1, 4.5],
                 },
             'mw': [350, 450, 550, 600]
            }
        substructure_matching: substructures to reward or penalize during biasing. Example:
            {
            'smiles': {'c1cccc1': 0.5},
            'smarts': {'[*]1[*][*][*][*][*]1': 0.1}
            }
        embedding_size: dimension of the embedding.
        hidden_size: dimension of the hidden layers.
        dropout_input: dropout applied to the embeddings before input to RNN.
        dropout_hidden: dropout applied between hidden layers of RNN.
        temperature: RNN temperature, alters diversity by rescaling logits.
        max_len: maximum sequence length that will be generated when sampling from the model.
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
    substructure_matching: Dict = field(default_factory=lambda: dict())

    embedding_size: int = 128
    hidden_size: int = 512
    dropout_input: float = 0
    dropout_hidden: float = 0
    temperature: float = 1.0

    max_len: int = 140

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
        embedding_size: dimension of the embedding.
        hidden_size: dimension of the hidden layers.
        dropout_input: dropout applied to the embeddings before input to RNN.
        dropout_hidden: dropout applied between hidden layers of RNN.
        num_epochs: number of training epochs.
        temperature: RNN temperature, alters diversity by rescaling logits.
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

    embedding_size: int = 128
    hidden_size: int = 512
    dropout_input: float = 0
    dropout_hidden: float = 0
    temperature: float = 1.0
