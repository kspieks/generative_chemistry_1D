import argparse
from pprint import pprint

import numpy as np
import pandas as pd
import umap
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.cluster import KMeans

from gen_chem_1D.data.data_classes import Postprocess
from gen_chem_1D.pred_models.features.featurizers import calc_morgan_fp
from gen_chem_1D.utils.parsing import read_yaml_file
from .similarity import get_top_N_most_similar


def standardize(smi):
    """Clean up SMILES string by neutralizing charges."""
    mol = Chem.MolFromSmiles(smi)
    clean_mol = rdMolStandardize.Cleanup(mol)
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    uncharger = rdMolStandardize.Uncharger()
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    return Chem.MolToSmiles(uncharged_parent_clean_mol)


def add_G_number(df):
    """Assigns a compound ID to the generated molecules with the format of G-xxxxxxx."""
    df.insert(2, 'Compound_ID', [f'G-{i:07}' for i in range(len(df))])
    return df


def calc_UMAP_clustering(df):
    """Determine UMAP projection and then cluster in the reduced dimension."""
    # calculate fingerprint representation
    df['fp'] = df['SMILES'].apply(calc_morgan_fp)
    fp_vectors = np.stack(df['fp'])
    # remove the temporary column
    df = df.drop('fp', axis=1)

    # define UMAP parameters and then transform the data
    umap_model = umap.UMAP(metric='jaccard',
                           n_neighbors=25,
                           n_components=2,
                           low_memory=False,
                           min_dist=0.001,
                           )
    X_umap = umap_model.fit_transform(fp_vectors)

    # add the projected coordinates to the df
    df['UMAP_0'] = X_umap[:, 0]
    df['UMAP_1'] = X_umap[:, 1]

    # arbitrarily create 20 clusters
    kmeans = KMeans(n_clusters=20,
                    random_state=0,
                    n_init='auto').fit(X_umap)
    df['kmeans_cluster'] = kmeans.labels_
    
    return df


def postprocess_data(postprocess_data_args):
    """
    Postprocess the generated SMILES.

    Args:
        postprocess_data_args: dataclass storing arguments for postprocessing the generated SMILES.
    """
    # if multiple rounds of generation were performed, concatenate all files
    dfs = []
    for file_path in postprocess_data_args.input_files:
        _df = pd.read_csv(file_path)
        dfs.append(_df)
    df = pd.concat(dfs)

    # remove any duplicates
    df = df.drop_duplicates(subset='inchi_key')

    if postprocess_data_args.neutralize:
        df['SMILES_original'] = df['SMILES'].copy()
        df['SMILES'] = df['SMILES'].apply(standardize)
        # recalculate InChI keys and remove any duplicates after standardizing
        df['inchi_key'] = df.SMILES.apply(lambda smi: Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
        df = df.drop_duplicates(subset='inchi_key')
    
    # remove any generated smiles that are already present in the training set
    df_train = pd.read_csv(postprocess_data_args.training_data, header=None)
    df_train.columns = ['SMILES']
    num_overlap = len(df[df.inchi_key.isin(df_train.inchi_key.values)])
    print(f'Removing {num_overlap} generated SMILES that are already present in the training set')
    df = df[~df.inchi_key.isin(df_train.inchi_key.values)]

    if postprocess_data_args.add_Gnum:
        print('Adding G-numbers...')
        df = add_G_number(df)
    
    if postprocess_data_args.add_UMAP_clustering:
        print('Performing UMAP clustering...')
        df = calc_UMAP_clustering(df)

    if postprocess_data_args.calc_top_N_most_similar:
        print('Calculating similarity w.r.t. closest training molecules...')
        df = get_top_N_most_similar(df, df_train,
                                    top_N=postprocess_data_args.top_N,
                                    ncpus=postprocess_data_args.num_cpus,
                                    )

    df.to_csv(postprocess_data_args.output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Script to process the final set of generated SMILES")
    parser.add_argument('--yaml_file', required=True,
                        help='Path to yaml file containing arguments for postprocessing.')
    args = parser.parse_args()

    # read yaml file
    print('Using arguments...')
    yaml_dict = read_yaml_file(args.yaml_file)
    pprint(yaml_dict)
    print('\n\n')

    # clean the generated data
    postprocess_data_args = Postprocess(**yaml_dict['postprocess_data'])
    postprocess_data(postprocess_data_args)


if __name__ == "__main__":
    main()