#!/usr/bin/env python

import argparse
from pprint import pprint

import numpy as np
from rdkit import Chem, RDLogger
import torch


from gen_chem_1D.gen_models.reinvent.data import Experience
from gen_chem_1D.gen_models.reinvent.model import RNN
from gen_chem_1D.gen_models.reinvent.tokenization import Vocabulary
from gen_chem_1D.gen_models.reinvent.utils import get_unique, seq_to_smiles, get_ss_score, validate_smiles, Variable
from gen_chem_1D.pred_models.scoring_functions import Scorer
from gen_chem_1D.data.data_classes import GenerativeBias
from gen_chem_1D.utils.parsing import read_yaml_file


def train_agent(gen_bias_args):
    """
    Biases a generative prior. 

    Args:
        gen_bias_args: dataclass storing arugments for biasing a generative prior.
    """
    # silence rdkit warnings
    RDLogger.DisableLog('rdApp.*') 

    if gen_bias_args.substructs:
        ss_frac = np.array([float(v) for v in gen_bias_args.substructs.values()])
        ss_patts = [Chem.MolFromSmiles(smi) for smi in gen_bias_args.substructs.keys()]

    # read in vocabulary and initialize prior
    voc = Vocabulary(init_from_file=gen_bias_args.vocab_file)
    Prior = RNN(voc=voc)
    Agent = RNN(voc=voc)

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
        seqs, _, _ = Agent.sample(batch_size=gen_bias_args.batch_size)
        Agent.rnn.train()
        agent_likelihood, entropy = Agent.likelihood(Variable(seqs))
        Agent.rnn.eval()

        # remove duplicates i.e., only consider unique sequences
        unique_idxs = get_unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # convert the generated sequence to smiles
        gen_smiles = seq_to_smiles(seqs, voc)
        
        # filter out any invalid and duplicate smiles
        valid_smiles, inchi_keys = validate_smiles(gen_smiles)
        inchi_keys, unique_idxs = np.unique(inchi_keys, return_index=True)

        seqs = seqs[unique_idxs]
        valid_unique_smiles = [valid_smiles[i] for i in unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # get prior likelihood, score, and fraction of acceptable
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        score, frac = scoring_function(valid_unique_smiles)

        if gen_bias_args.substructs:
            score_sub = get_ss_score(valid_unique_smiles, ss_patts)
            print(f'Num with Substruct: {[str(int(si)) for si in np.sum(score_sub, axis=0)]}')
            score = score * (1 + np.sum(ss_frac * score_sub, axis=1))

        # calculate augmented likelihood and loss
        augmented_likelihood = prior_likelihood + gen_bias_args.reward_multiplier * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(valid_unique_smiles, score, prior_likelihood)
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
        print(f'Generated {len(valid_smiles)} valid SMILES i.e., {len(valid_smiles)/gen_bias_args.batch_size * 100:.2f}%')
        print(f'From those, {len(valid_unique_smiles)} were unique i.e., {len(valid_unique_smiles)/len(valid_smiles) * 100:.2f}%')
        print('Fraction in acceptable range:')
        for i, n in enumerate(names):
            print(f'{n}: {frac[i]:.3f}')

        print(f"Agent LL: {np.mean(agent_likelihood):6.2f} Prior LL: {np.mean(prior_likelihood):6.2f} Aug LL: {np.mean(augmented_likelihood):6.2f} Score: {np.mean(score):6.2f}")
        print("  Agent  Prior   Target  Score       SMILES")
        for i in range(min(10, len(valid_smiles) )):
            print(f"{agent_likelihood[i]:6.2f}\t{prior_likelihood[i]:6.2f}\t{augmented_likelihood[i]:6.2f}\t{score[i]:6.2f} {valid_smiles[i]}")
        print('\n\n')

        # save this agent in case we want to go back to it
        torch.save(Agent.rnn.state_dict(), f'gen_model/biased_agent_step_{step}.ckpt')
        if np.mean(score) > best_score:
            best_score = np.mean(score)
            # also update the checkpoint for the best agent that satisfies the most objectives
            torch.save(Agent.rnn.state_dict(), 'gen_model/biased_agent.ckpt') 


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
