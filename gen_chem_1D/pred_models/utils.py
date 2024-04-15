"""
Utility functions for evaluating regression models.
"""
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def calc_regression_metrics(y_true, y_pred, ranking_metrics=True):
    """
    Calculate performance metrics for the regression model.
    The naive baseline model used in this codebase naively predicts
    a constant value for the y_pred vector so for this case, it does not
    make sense to report kendall tau or spearman rank metrics.

    Args:
        y_true: np.array of true values.
        y_pred: np.array of values predicted from a regression model.
        ranking_metrics: boolean indicating whether to return ranking metrics.

    Returns:
        metrics: dictionary with performance metrics.
    """
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = root_mean_squared_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    metrics = {
        'MAE': MAE,
        'RMSE': RMSE,
        'R2': R2
    }

    if ranking_metrics:
        kendalltau = stats.kendalltau(y_true, y_pred)
        metrics['kendall_tau_statistic'] = kendalltau.statistic
        metrics['kendall_tau_pvalue'] = kendalltau.pvalue

        spearman = stats.spearmanr(y_true, y_pred)
        metrics['spearman_statistic'] = spearman.statistic
        metrics['spearman_pvalue'] = spearman.pvalue
    
    for key, value in metrics.items():
        if 'pvalue' in key:
            print(f'{key}: {value:.4e}')
        else:
            print(f'{key}: {value:.4f}')

    return metrics


def naive_baseline(y_train, y_test, y_val=None):
    """
    Use a naive baseline model that simply predicts the mean value
    from the training set for all test molecules. This is helpful
    to understand the inherent variance in the data.
    """
    print(f'Mean target value for training set: {y_train.mean():.3f}')
    if y_val:
        print(f'Mean target value for validation set: {y_val.mean():.3f}')
    print(f'Mean target value for testing set: {y_test.mean():.3f}\n')

    # naive baseline model uses mean of the training set
    y_train_pred = [y_train.mean()] * len(y_train)
    print('Performance of naive baseline model on the training set:')
    training_metrics = calc_regression_metrics(y_train, y_train_pred, ranking_metrics=False)

    if y_val:
        y_val_pred = [y_train.mean()] * len(y_val)
        print('\nPerformance of naive baseline model on the validation set:')
        validation_metrics = calc_regression_metrics(y_val, y_val_pred, ranking_metrics=False)

    y_test_pred = [y_train.mean()] * len(y_test)
    print('\nPerformance of naive baseline model on the testing set:')
    testing_metrics = calc_regression_metrics(y_test, y_test_pred, ranking_metrics=False)
