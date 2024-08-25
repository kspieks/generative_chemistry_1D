import argparse
from pprint import pprint

import pandas as pd

from gen_chem_1D.data.data_classes import (GenerativeBias, GenerativePrior,
                                           GenerativeSample, PredictiveModel,
                                           Preprocess)
from gen_chem_1D.gen_models.reinvent.preprocess_gen_data import clean_gen_data
from gen_chem_1D.gen_models.reinvent.sample import sample
from gen_chem_1D.gen_models.reinvent.train_agent import train_agent
from gen_chem_1D.gen_models.reinvent.train_prior import train_prior
from gen_chem_1D.pred_models.preprocess_pred_data import clean_pred_data
from gen_chem_1D.pred_models.train_pred_models import train_pred_model
from gen_chem_1D.utils.parsing import read_yaml_file


def main():
    parser = argparse.ArgumentParser(description="Script to run end-to-end workflow.")
    parser.add_argument('--yaml_file', required=True,
                        help='Path to yaml file containing arguments.')
    args = parser.parse_args()

    # read yaml file
    print('Using arguments...')
    yaml_dict = read_yaml_file(args.yaml_file)
    pprint(yaml_dict)
    print('\n\n')

    # clean and filter the data for predictive modeling
    preprocess_data_args = Preprocess(**yaml_dict['preprocess_data'])
    clean_pred_data(preprocess_data_args)

    # train predictive model(s)
    pred_model_args = PredictiveModel(**yaml_dict['pred_model'])
    train_pred_model(pred_model_args)


    # clean and filter the data for generative modeling
    preprocess_data_args = Preprocess(**yaml_dict['preprocess_data'])
    clean_gen_data(preprocess_data_args)

    # train generative prior
    gen_prior_args = GenerativePrior(**yaml_dict['gen_model']['pre_train'])
    train_prior(gen_prior_args)

    # sample from the generative model before biasing
    gen_sample_args = GenerativeSample(**yaml_dict['gen_model']['sample'])
    gen_sample_args.checkpoint_path = 'gen_model/Prior.ckpt'
    gen_sample_args.output_file = 'gen_model/generated_smiles_prior.csv'
    sample(gen_sample_args)


    # bias generative prior
    gen_bias_args = GenerativeBias(**yaml_dict['gen_model']['bias'])
    train_agent(gen_bias_args)

    # sample from the biased generative model
    gen_sample_args.checkpoint_path = 'gen_model/Agent.ckpt'
    gen_sample_args.output_file = 'gen_model/generated_smiles_agent.csv'
    gen_sample_args.num_smiles = 500
    sample(gen_sample_args)

if __name__ == "__main__":
    main()
