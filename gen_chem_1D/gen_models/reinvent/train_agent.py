#!/usr/bin/env python

import argparse
import os
from pprint import pprint

import numpy as np
import torch
from rdkit import Chem, RDLogger

from gen_chem_1D.data.data_classes import GenerativeBias
from gen_chem_1D.gen_models.reinvent.data import Experience
from gen_chem_1D.gen_models.reinvent.model import RNN
from gen_chem_1D.gen_models.reinvent.tokenization import Vocabulary
from gen_chem_1D.gen_models.reinvent.utils import (Variable, get_ss_score,
                                                   get_unique_indices, seq_to_smiles,
                                                   get_valid_unique_smiles_idx)
from gen_chem_1D.pred_models.scoring_functions import Scorer
from gen_chem_1D.utils.parsing import read_yaml_file


def train_agent(gen_bias_args):
    """
    Biases a generative prior. 

    Args:
        gen_bias_args: dataclass storing arugments for biasing a generative prior.
    """
    print('Biasing generative model...')
    # silence rdkit warnings
    RDLogger.DisableLog('rdApp.*') 

    if gen_bias_args.substructure_matching:
        ss_frac = []
        ss_patts = []
        for key, sub_dict in gen_bias_args.substructure_matching.items():
            if key == 'smiles':
                ss_frac.extend([float(v) for v in sub_dict.values()])
                ss_patts.extend([Chem.MolFromSmiles(smi) for smi in sub_dict.keys()])
            elif key == 'smarts':
                ss_frac.extend([float(v) for v in sub_dict.values()])
                ss_patts.extend([Chem.MolFromSmarts(smi) for smi in sub_dict.keys()])
        ss_frac = np.array(ss_frac)

    # read in vocabulary and initialize prior
    voc = Vocabulary(init_from_file=gen_bias_args.vocab_file)
    Prior = RNN(voc=voc,
                embedding_size=gen_bias_args.embedding_size,
                hidden_size=gen_bias_args.hidden_size,
                dropout_input=gen_bias_args.dropout_input,
                dropout_hidden=gen_bias_args.dropout_hidden,
                temperature=gen_bias_args.temperature,
                )
    Agent = RNN(voc=voc,
                embedding_size=gen_bias_args.embedding_size,
                hidden_size=gen_bias_args.hidden_size,
                dropout_input=gen_bias_args.dropout_input,
                dropout_hidden=gen_bias_args.dropout_hidden,
                temperature=gen_bias_args.temperature,
                )

    # saved models are partially on the GPU, but if we don't have cuda enabled we can re-map these to the CPU
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(gen_bias_args.prior_checkpoint_path))
        Agent.rnn.load_state_dict(torch.load(gen_bias_args.agent_checkpoint_path))
    else:
        Prior.rnn.load_state_dict(torch.load(gen_bias_args.prior_checkpoint_path, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(gen_bias_args.agent_checkpoint_path, map_location=lambda storage, loc: storage))
    Prior.rnn.eval()
    Agent.rnn.eval()

    # don't need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=gen_bias_args.init_lr)

    # define scoring function(s)
    scoring_function = Scorer(scoring_functions=gen_bias_args.scoring_functions)
    names = scoring_function.names

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    print("Model initialized, starting training...")
    best_score = 0
    save_limit = 2
    save_paths = []
    for step in range(gen_bias_args.num_steps):
        # increase learning rate linearly from 5% to 100% of specified rate over first 20 steps
        if step < 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = (step + 1) * gen_bias_args.init_lr / 20.0

        # after 20 steps, decrease learning rate by 2% for each step
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.98

        # sample from Agent
        seqs, _, _ = Agent.sample(batch_size=gen_bias_args.batch_size,  max_length=gen_bias_args.max_len)
        Agent.rnn.train()
        agent_likelihood, entropy = Agent.likelihood(Variable(seqs))
        Agent.rnn.eval()

        # remove duplicates i.e., only consider unique sequences
        unique_idxs = get_unique_indices(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # convert the generated sequences to a list of smiles
        gen_smiles = seq_to_smiles(seqs, voc)
        
        # filter out any invalid and duplicate smiles
        v_smiles, v_inchi_key, vu_smiles, vu_inchi_key, vu_indices, ndup = get_valid_unique_smiles_idx(gen_smiles)
        seqs = seqs[vu_indices]
        agent_likelihood = agent_likelihood[vu_indices]
        entropy = entropy[vu_indices]

        # get prior likelihood, score, and fraction of acceptable
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        score, frac = scoring_function(vu_smiles)

        if gen_bias_args.substructure_matching:
            score_sub = get_ss_score(vu_smiles, ss_patts)
            print(f'Num with substruct: {[str(int(si)) for si in np.sum(score_sub, axis=0)]}')
            score = score * (1 + np.sum(ss_frac * score_sub, axis=1))

        # calculate augmented likelihood and loss
        augmented_likelihood = prior_likelihood + gen_bias_args.reward_multiplier * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        # ensure that dimensions all match before zipping together
        if len(vu_smiles) != len(score) or len(score) != len(prior_likelihood):
            msg = 'Dimension mismatch!\n'
            msg += f'vu_smiles has {len(vu_smiles)} entries\n'
            msg += f'score has {len(score)} entries\n'
            msg += f'prior_likelihood has {len(prior_likelihood)} entries\n'
            raise ValueError(msg)
        new_experience = zip(vu_smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        loss = loss.mean()

        # add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        print(f'Step {step}: Loss = {loss.item():.2f}')
        print(f'Generated {len(v_smiles)} valid SMILES i.e., {len(v_smiles)/gen_bias_args.batch_size * 100:.2f}%')
        print(f'{len(vu_smiles)} were unique i.e., {len(vu_smiles)/len(v_smiles) * 100:.2f}% of the valid SMILES and {len(vu_smiles)/gen_bias_args.batch_size * 100:.2f}% of the batch')
        print(f'{ndup} generated molecules had duplicate InChi keys')
        print('Fraction in acceptable range:')
        for i, n in enumerate(names):
            print(f'{n}: {frac[i]:.3f}')

        print(f"Agent LL: {np.mean(agent_likelihood):6.2f} Prior LL: {np.mean(prior_likelihood):6.2f} Aug LL: {np.mean(augmented_likelihood):6.2f} Score: {np.mean(score):6.2f}")
        print("Agent\tPrior\tTarget\tScore\tSMILES")
        for i in range(min(10, len(vu_smiles) )):
            print(f"{agent_likelihood[i]:6.2f}\t{prior_likelihood[i]:6.2f}\t{augmented_likelihood[i]:6.2f}\t{score[i]:6.2f} {vu_smiles[i]}")
        print('\n\n')

        if np.mean(score) > best_score:
            best_score = np.mean(score)
            # update the checkpoint for the best agent that satisfies the most objectives
            # delete the worst model from the list of best models
            if len(save_paths) >= save_limit:
                path_to_delete = save_paths.pop(0)
                os.remove(path_to_delete)
            save_path = f'gen_model/biased_agent_step_{step}.ckpt'
            torch.save(Agent.rnn.state_dict(), save_path)
            save_paths.append(save_path)
    
    # save the final trained Agent
    torch.save(Agent.rnn.state_dict(), gen_bias_args.agent_save_path) 


def main():
    parser = argparse.ArgumentParser(description="Script to train generative prior.")
    parser.add_argument('--yaml_file', required=True,
                        help='Path to yaml file containing arguments for training a generative prior.')
    args = parser.parse_args()

    # read yaml file
    print('Using arguments...')
    yaml_dict = read_yaml_file(args.yaml_file)
    pprint(yaml_dict)
    print('\n\n')
    
    # bias generative prior
    gen_bias_args = GenerativeBias(**yaml_dict['gen_model']['bias'])
    train_agent(gen_bias_args)


if __name__ == "__main__":
    main()
