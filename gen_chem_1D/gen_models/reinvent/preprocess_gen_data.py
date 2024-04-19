import argparse
import os
from pprint import pprint

import pandas as pd

from gen_chem_1D.data.data_cleaning import clean_smiles
from gen_chem_1D.data.data_classes import Preprocess
from gen_chem_1D.gen_models.reinvent.tokenization import create_vocabulary
from gen_chem_1D.utils.parsing import read_yaml_file


def clean_gen_data(preprocess_data_args):
    """
    Cleans data in preparation for training generative model(s).

    Args:
        preprocess_data_args: dataclass storing arguments for preprocessing the SMILES.
    """

    # read in data as df
    df = pd.read_csv(preprocess_data_args.pred_input_file)

    # clean the SMILES
    df = clean_smiles(df,
                      min_heavy_atoms=preprocess_data_args.min_heavy_atoms,
                      max_heavy_atoms=preprocess_data_args.max_heavy_atoms,
                      supported_elements=preprocess_data_args.supported_elements,
                      )
    
    # save the cleaned SMILES
    directory = os.path.dirname(preprocess_data_args.gen_output_file)
    os.makedirs(directory, exist_ok=True)
    with open(preprocess_data_args.gen_output_file, 'w') as f:
        f.write('\n'.join(df.SMILES.values) + '\n')

    # create the vocabulary
    vocab_tokens = create_vocabulary(df.SMILES.values)
    print(f'Vocabulary contains {len(vocab_tokens)} tokens:')
    print('\n'.join(vocab_tokens))

    # save the vocab tokens
    vocab_tokens = list(vocab_tokens)
    vocab_tokens.sort()
    with open(preprocess_data_args.gen_vocab_file, 'w') as f:
        for char in vocab_tokens:
            f.write(char + "\n")


def main():
    parser = argparse.ArgumentParser(description="Script to process data in preparation for training generative model(s).")
    parser.add_argument('--yaml_file', required=True,
                        help='Path to yaml file containing arguments for preprocessing.')
    args = parser.parse_args()

    # read yaml file
    print('Using arguments...')
    yaml_dict = read_yaml_file(args.yaml_file)
    pprint(yaml_dict)
    print('\n\n')
    
    # clean and filter the data
    preprocess_data_args = Preprocess(**yaml_dict['preprocess_data'])
    clean_gen_data(preprocess_data_args)


if __name__ == "__main__":
    main()
