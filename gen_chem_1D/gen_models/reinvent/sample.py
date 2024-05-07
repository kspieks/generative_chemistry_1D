#!/usr/bin/env python

import argparse
from pprint import pprint

import numpy as np
import pandas as pd
from rdkit import RDLogger
import torch

from gen_chem_1D.data.data_classes import GenerativeSample
from gen_chem_1D.gen_models.reinvent.model import RNN, ScaffoldConstrainedRNN
from gen_chem_1D.gen_models.reinvent.tokenization import Vocabulary
from gen_chem_1D.gen_models.reinvent.utils import seq_to_smiles, validate_smiles
from gen_chem_1D.pred_models.rf_model import predict_rf
from gen_chem_1D.utils.parsing import read_yaml_file


def sample(gen_sample_args):
    """
    Samples from a generative model.

    Args:
        gen_sample_args: dataclass storing arugments for sampleing from a generative model.
    """
    # silence rdkit warnings
    RDLogger.DisableLog('rdApp.*')  

    # load vocabulary and initialize generator
    voc = Vocabulary(init_from_file=gen_sample_args.vocab_file)
    if gen_sample_args.scaffold_constraint:
        Agent = ScaffoldConstrainedRNN(voc=voc)
    else:
        Agent = RNN(voc=voc)

    # put model on device and set to evaluation mode
    if torch.cuda.is_available():
        Agent.rnn.load_state_dict(torch.load(gen_sample_args.checkpoint_path))
    else:
        Agent.rnn.load_state_dict(torch.load(gen_sample_args.checkpoint_path, map_location=lambda storage, loc: storage))
    Agent.rnn.eval()

    # don't need gradients for sampling
    with torch.no_grad():
        gen_valid_unique_smiles = []
        gen_inchi_keys = []
        step=0
        print("Model initialized, starting sampling...")
        print(f'Each batch will sample {gen_sample_args.batch_size} generated sequences')
        while len(gen_valid_unique_smiles) < gen_sample_args.num_smiles:
            step += 1   # count number of iterations
            print(f'Batch {step}:')

            # sample from Agent and convert the generated sequence to smiles
            if gen_sample_args.scaffold_constraint:
                seqs, _agent_likelihood, _entropy = Agent.sample(gen_sample_args.batch_size,
                                                                 pattern=gen_sample_args.scaffold_constraint,
                                                                 max_length=gen_sample_args.max_len)
            else:
                seqs, _agent_likelihood, _entropy = Agent.sample(gen_sample_args.batch_size,
                                                                 max_length=gen_sample_args.max_len)
            gen_smiles = seq_to_smiles(seqs, voc)

            # filter out any invalid and duplicate smiles
            valid_smiles, inchi_keys = validate_smiles(gen_smiles)
            inchi_keys, idxs = np.unique(inchi_keys, return_index=True)
            valid_unique_smiles = [valid_smiles[i] for i in idxs]
            gen_valid_unique_smiles += valid_unique_smiles
            gen_inchi_keys += list(inchi_keys)
            print(f'Generated {len(valid_smiles)} valid SMILES i.e., {len(valid_smiles)/gen_sample_args.batch_size * 100:.2f}%')
            print(f'From those, {len(valid_unique_smiles)} were unique i.e., {len(valid_unique_smiles)/len(valid_smiles) * 100:.2f}%\n')
            print(f'Total generated: {len(gen_valid_unique_smiles)}')

            # remove any duplcates
            if len(gen_valid_unique_smiles) >= gen_sample_args.num_smiles:
                gen_inchi_keys, idxs = np.unique(gen_inchi_keys, return_index=True)
                gen_valid_unique_smiles = [gen_valid_unique_smiles[i] for i in idxs]
                gen_inchi_keys = list(gen_inchi_keys)
    
    percent_vu = len(gen_valid_unique_smiles)/(step * gen_sample_args.batch_size)*100
    print(f'In total, {len(gen_valid_unique_smiles)} SMILES (i.e., {percent_vu:.2f}%) were valid and unique')

    df = pd.DataFrame(gen_valid_unique_smiles, columns=['SMILES'])
    df['inchi_key'] = gen_inchi_keys

    # get scores for the generated smiles
    for pred_target, sub_dict in gen_sample_args.pred_models.items():
        if sub_dict['model'] == 'random_forest':
            print(f'Obtaining predictions for {pred_target}')
            preds = predict_rf(gen_valid_unique_smiles, sub_dict['path'])
            df[f'rf_{pred_target}'] = preds

    df.to_csv(gen_sample_args.output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Script to sample from a generative model.")
    parser.add_argument('--yaml_file', required=True,
                        help='Path to yaml file containing arguments for sampling from a generative prior.')
    args = parser.parse_args()

    # read yaml file
    print('Using arguments...')
    yaml_dict = read_yaml_file(args.yaml_file)
    pprint(yaml_dict)
    print('\n\n')
    
    # sample from the generative model
    gen_sample_args = GenerativeSample(**yaml_dict['gen_model']['sample'])
    sample(gen_sample_args)


if __name__ == "__main__":
    main()
