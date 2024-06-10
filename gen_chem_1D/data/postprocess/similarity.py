import os
import pickle as pkl

from joblib import Parallel, delayed
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


fp_gen = AllChem.GetMorganGenerator(radius=3, includeChirality=True, fpSize=2048)
def get_morgan_bit_fp(smi, fp_gen=fp_gen):
    mol = Chem.MolFromSmiles(smi)
    return fp_gen.GetFingerprint(mol)


def get_sim(fp, fp_list):
    similarities = DataStructs.BulkTanimotoSimilarity(fp, fp_list)
    return similarities


def get_similarity_matrix(fp_list1, fp_list2, ncpus=1):
    """
    Returns a matrix of Tanimoto similarities as a numpy array
    of size fp_list1 x fp_list2.

	Progress is measured by the number of smiles in fp_list1.
    """
    result = Parallel(n_jobs=ncpus, backend='multiprocessing', verbose=5)(delayed(get_sim)(fp, fp_list2) for fp in fp_list1)

    similarity_matrix = np.zeros((len(fp_list1), len(fp_list2)))
    for i, res in enumerate(result):
        similarity_matrix[i, :] = res
    
    return similarity_matrix


def get_top_N_most_similar(df_gen, df_train, top_N=3, ncpus=1):
    # get list of rdkit.DataStructs.UIntSparseIntVect fingerprints
    print('Calculating fingerprints...')
    df_gen['fp'] = df_gen['SMILES'].apply(get_morgan_bit_fp)
    df_train['fp'] = df_train['SMILES'].apply(get_morgan_bit_fp)

    fp_list1 = df_gen.fp.values
    fp_list2 = df_train.fp.values

    # remove the temporary column
    df_gen = df_gen.drop('fp', axis=1)
    df_train = df_train.drop('fp', axis=1)

    print(f'Generated set has {len(fp_list1)} SMILES')
    print(f'Training set has {len(fp_list2)} SMILES')
    print('Progress is measured by the number of molecules in the generated set')
    print('Calculating similarity matrix...')
    similarity_matrix = get_similarity_matrix(fp_list1, fp_list2, ncpus=ncpus)

    # save the matrix so the user doesn't have to compute it again
    d0, d1 = similarity_matrix.shape
    with open(f'similarity_matrix_{d0}x{d1}.pkl', 'wb') as f:
        pkl.dump(similarity_matrix, f)
    
    # get the top N most similar matches from the training set
    num_identical = len(np.argwhere(similarity_matrix == 1))
    print(f'{num_identical} matches had similarity of 1.0')
    training_smiles = df_train['SMILES'].values
    smi_dict = {f'match{i+1}_smi': [] for i in range(top_N)}
    sim_dict = {f'match{i+1}_sim': [] for i in range(top_N)}

    for i in range(similarity_matrix.shape[0]):
        temp_sim_array = np.array(similarity_matrix[i, :])
        # argsort returns the indices that would sort the array from smallest to largest
        # so take the last N of those but then reverse the list to get most similar to least similar
        indices = temp_sim_array.argsort()[-top_N:][::-1]

        for jj, idx in enumerate(indices):
            smi_dict[f'match{jj+1}_smi'].append(training_smiles[idx])
            sim_dict[f'match{jj+1}_sim'].append(temp_sim_array[idx])

    # put the training smiles and corresponding similarity in the generated df
    for i in range(top_N):
        df_gen[f'match{i+1}_smi'] = smi_dict[f'match{i+1}_smi']
        df_gen[f'match{i+1}_sim'] = smi_dict[f'match{i+1}_sim']

    return df_gen
