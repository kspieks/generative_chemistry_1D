import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features.create_features import create_features
from .utils import calc_regression_metrics, naive_baseline

RF_HYPERPARAMETERS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42
}

def train_rf(df,
             regression_targets,
             save_dir='pred_models',
             hyperparameters=RF_HYPERPARAMETERS,
             ):
    """
    Trains a random forest regressor for each target value.

    Args:
        df: dataframe containing a column of SMILES and column(s) of target(s).
        regression_targets: list of regression targets to fit a RF model to.
        save_dir: directory to save model and prediction results to.
        hyperparameters: dictionary of hyperparameters to define the RF model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for target in regression_targets:
        # remove any molecules with missing values
        df_target = df[~df[target].isna()]

        # create molecular features
        X = create_features(df_target)
        y = df_target[target].values

        # use 80:20 time split to evaluate model performance
        print(f'\n\n{target} has {len(df_target)} data points in total')
        num_train = int(0.8 * len(y))
        num_test = len(y) - num_train
        print('After splitting the data into 80% training and 20% testing, there are:')
        print(f'{num_train} training examples')
        print(f'{num_test} testing examples')

        X_train = X[:num_train, :]
        y_train = y[:num_train]

        X_test = X[num_train:, :]
        y_test = y[num_train:]

        # first train a naive baseline model to understand the inherent variance in the data
        print('\nTraining naive baseline model that simply predicts the mean of the training set...')
        naive_baseline(y_train, y_test)

        # now train a random forest model
        print('\n\nTraining random forest model...')
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
        df_preds = pd.DataFrame(df_target.SMILES.values, columns=['SMILES'])
        split_label = ['train'] * num_train
        split_label.extend(['test'] * num_test)
        df_preds['split'] = split_label

        df_preds[f'{target}_true'] = y
        df_preds[f'{target}_pred'] = np.concatenate((y_train_pred, y_test_pred))

        # re-train model using all data
        rf_model.fit(X, y)

        # save the extended model
        pkl_file = os.path.join(save_dir, f'rf_{target}.pkl')
        with open(pkl_file, 'wb') as f:
            pkl.dump(rf_model, f)
    
        # save the test set predictions so we can make plots later
        csv_file = os.path.join(save_dir, f'rf_{target}_preds.csv')
        df_preds.to_csv(csv_file, index=False)
