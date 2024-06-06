import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from gen_chem_1D.data.data_classes import RF_HYPERPARAMETERS
from .features.create_features import create_features
from .utils import calc_regression_metrics, naive_baseline


def train_rf(df,
             regression_targets,
             save_dir='pred_models',
             hyperparameters=RF_HYPERPARAMETERS,
             split='random',
             ):
    """
    Trains a random forest regressor for each target value.

    Args:
        df: dataframe containing a column of SMILES and column(s) of target(s).
        regression_targets: list of regression targets to fit a RF model to.
        save_dir: directory to save model and prediction results to.
        hyperparameters: dictionary of hyperparameters to define the RF model.
        split: string indicating how to evaluate the RF model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for target in regression_targets:
        # remove any molecules with missing values
        df_target = df[~df[target].isna()].reset_index(drop=True)
        print('*'*88)
        print(f'{target} has {len(df_target)} data points in total')

        # create molecular features
        print('Creating features...')
        X = create_features(df_target)
        y = df_target[target].values

        if split == 'random':
            # use 80:20 random split to evaluate model performance
            print('Randomly splitting the data into 80% training and 20% testing...')
            df_train = df_target.sample(frac=0.8)
            df_test = df_target.drop(df_train.index)
            num_train = len(df_train)
            num_test = len(df_test)
        elif split == 'time_split':
            # use 80:20 time split to evaluate model performance
            print('Splitting the data into 80% training and 20% testing via time split...')
            num_train = int(0.8 * len(y))
            num_test = len(y) - num_train
            df_train = df_target[:num_train]
            df_test = df_target[num_train:]
        
        print(f'{num_train} training examples')
        print(f'{num_test} testing examples')

        X_train = X[df_train.index.values, :]
        y_train = y[df_train.index.values]

        X_test = X[df_test.index.values, :]
        y_test = y[df_test.index.values]

        # first train a naive baseline model to understand the inherent variance in the data
        print('\nTraining naive baseline model that simply predicts the mean of the training set...')
        naive_baseline(y_train, y_test)

        # now train a random forest model
        print('\nTraining random forest model...')
        rf_model = RandomForestRegressor(**hyperparameters)
        rf_model.fit(X_train, y_train)
        
        # evaluate the performance on the training set
        print('Performance of RF model on the training set:')
        y_train_pred = rf_model.predict(X_train)
        training_metrics = calc_regression_metrics(y_train, y_train_pred, ranking_metrics=True)

        # evaluate the performance on the testing set
        print('\nPerformance of RF model on the testing set:')
        y_test_pred = rf_model.predict(X_test)
        testing_metrics = calc_regression_metrics(y_test, y_test_pred, ranking_metrics=True)

        # create a df to store the prediction results from the train test split
        SMILES = list(df_train.SMILES.values)
        SMILES.extend(df_test.SMILES.values)
        df_preds = pd.DataFrame(SMILES, columns=['SMILES'])

        split_label = ['train'] * num_train
        split_label.extend(['test'] * num_test)
        df_preds['split'] = split_label

        df_preds[f'{target}_true'] = np.concatenate((y_train, y_test))
        df_preds[f'{target}_pred'] = np.concatenate((y_train_pred, y_test_pred))

        # re-train model using all data
        print('\nRetraining RF model on all data...')
        rf_model.fit(X, y)

        # save the extended model
        pkl_file = os.path.join(save_dir, f'rf_{target}.pkl')
        print(f'Saving the RF model to {pkl_file}\n')
        with open(pkl_file, 'wb') as f:
            pkl.dump(rf_model, f)
    
        # save the predictions so we can make plots later
        csv_file = os.path.join(save_dir, f'rf_{target}_preds.csv')
        df_preds.to_csv(csv_file, index=False)


def predict_rf(smiles, file_path):
    # put SMILES in a dataframe
    df = pd.DataFrame(smiles, columns=['SMILES'])

    # create molecular features
    X = create_features(df)

    # get the trained model and make predictions
    with open(file_path, 'rb') as f:
        rf_model = pkl.load(f)
    y_pred = rf_model.predict(X)

    return y_pred
