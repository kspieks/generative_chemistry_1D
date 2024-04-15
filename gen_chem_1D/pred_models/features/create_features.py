import numpy as np

from .featurizers import calc_atompair_fp, calc_donorpair_fp


def create_features(df):
    """
    Calculates features for SMILES strings in a pd.Dataframe.

    Args:
        df: pandas dataframe which contains a column of SMILES strings.
    
    Returns:
        X: np.array of shape N molecules x 2048 features.
           The feature vector length is hardcoded for now.
    """
    # get atom pair fingerprints
    df_ap = df.SMILES.apply(calc_atompair_fp)
    ap_fps = np.stack(df_ap.values)

    # get donor pair fingerprints
    df_dp = df.SMILES.apply(calc_donorpair_fp)
    dp_fps = np.stack(df_dp.values)

    X = np.hstack([ap_fps, dp_fps])

    return X
