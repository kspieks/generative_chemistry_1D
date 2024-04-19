#!/usr/bin/env python

import argparse
import os
from pprint import pprint

import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_chem_1D.data.data_classes import GenerativePrior
from gen_chem_1D.gen_models.reinvent.data import MolData
from gen_chem_1D.gen_models.reinvent.model import RNN
from gen_chem_1D.gen_models.reinvent.tokenization import Vocabulary
from gen_chem_1D.gen_models.reinvent.utils import decrease_learning_rate
from gen_chem_1D.utils.parsing import read_yaml_file


def train_prior(gen_prior_args):
    """
    Trains a generative prior. 

    Args:
        gen_prior_args: dataclass storing arugments for training a generative prior.
    """
    # silence rdkit warning
    RDLogger.DisableLog('rdApp.*') 

    # create output directory
    os.makedirs(gen_prior_args.out_dir, exist_ok=True)

    # read in vocabulary and initialize prior
    voc = Vocabulary(init_from_file=gen_prior_args.vocab_file)
    Prior = RNN(voc=voc)

    # optionally restore and continue training an RNN
    if gen_prior_args.checkpoint_path:
        Prior.rnn.load_state_dict(torch.load(gen_prior_args.checkpoint_path))
    
    # set model to train mode
    Prior.rnn.train()

    # create a PyTorch Dataset from a SMILES file
    moldata = MolData(gen_prior_args.smi_path, voc)
    data = DataLoader(moldata, batch_size=gen_prior_args.batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)
    
    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=gen_prior_args.init_lr)
    for epoch in range(1, gen_prior_args.num_epochs):
        print(f'Epoch: {epoch}')
        for step, batch in tqdm(enumerate(data), total=len(data)):
            # sample from DataLoader
            seqs = batch.long()

            # calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = -log_p.mean()

            # calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # every N steps, decrease learning rate and print some information
            N = 50
            if (step % N == 0 and step != 0) or (len(data) < N and step==len(data) - 1):
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write(f"Epoch {epoch:4d}\t step {step:4d}\t loss: {loss.data.item():5.2f}\n")

                # test the validity of the generated smiles
                Prior.rnn.eval()
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                percent_valid = valid / len(seqs) * 100
                tqdm.write(f"\n{percent_valid:>4.1f}% valid SMILES")
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), os.path.join(gen_prior_args.out_dir, f"Prior_epoch_{epoch}.ckpt"))

                Prior.rnn.train()
        
    # save the trained prior
    torch.save(Prior.rnn.state_dict(), os.path.join(gen_prior_args.out_dir, "Prior.ckpt"))


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
    
    # train generative prior
    gen_prior_args = GenerativePrior(**yaml_dict['gen_model']['pre_train'])
    train_prior(gen_prior_args)


if __name__ == "__main__":
    main()
