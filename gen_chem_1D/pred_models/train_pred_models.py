import argparse
from pprint import pprint

import pandas as pd

from gen_chem_1D.pred_models.rf_model import train_rf
from gen_chem_1D.data.data_classes import PredictiveModel
from gen_chem_1D.utils.parsing import read_yaml_file


def train_pred_model(pred_model_args):
    """
    Trains predictive model(s) for the desired regression targets.

    Args:
        pred_model_args: dataclass storing arugments for training predictive model(s).
    """

    # read in data as df
    df = pd.read_csv(pred_model_args.input_file)

    # train the predictive model(s)
    print('Training predictive models...')
    if pred_model_args.model == 'random_forest':
        train_rf(df,
                 regression_targets=pred_model_args.regression_targets,
                 save_dir=pred_model_args.save_dir,
                 hyperparameters=pred_model_args.hyperparameters
                 )
    else:
        msg = f"Currently, `random_forest` is the only supported model type. Got: {pred_model_args.model}."
        raise ValueError(msg)


def main():
    parser = argparse.ArgumentParser(description="Script to train predictive models.")
    parser.add_argument('--yaml_file', required=True,
                        help='Path to yaml file containing arguments for training predictive model(s).')
    args = parser.parse_args()

    # read yaml file
    print('Using arguments...')
    yaml_dict = read_yaml_file(args.yaml_file)
    pprint(yaml_dict)
    print('\n\n')
    
    # train predictive model(s)
    pred_model_args = PredictiveModel(**yaml_dict['pred_model'])
    train_pred_model(pred_model_args)


if __name__ == "__main__":
    main()
