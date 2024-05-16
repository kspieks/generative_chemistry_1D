import argparse
import os
from pprint import pprint

import pandas as pd

from gen_chem_1D.data.data_classes import Preprocess
from gen_chem_1D.data.data_cleaning import clean_smiles
from gen_chem_1D.utils.parsing import read_yaml_file


def clean_pred_data(preprocess_data_args):
    """
    Cleans data in preparation for training predictive model(s).

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
                      remove_stereo=preprocess_data_args.remove_stereo,
                      canonicalize=preprocess_data_args.canonicalize,
                      )
    
    output_dir = os.path.dirname(preprocess_data_args.pred_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(preprocess_data_args.pred_output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Script to process data in preparation for training predictive model(s).")
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
    clean_pred_data(preprocess_data_args)

if __name__ == "__main__":
    main()
