"""Decoding functions."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from typing import Optional

def build_feature_index(
    acts_df_train: pd.DataFrame,
    acts_df_val: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create a consistent column index for features across train/val.
    Returns a DataFrame with columns: ['instance_idx','latent_idx','col'].
    """
    frames = [acts_df_train[['instance_idx','latent_idx']].drop_duplicates()]
    if acts_df_val is not None:
        frames.append(acts_df_val[['instance_idx','latent_idx']].drop_duplicates())
    pairs = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(['instance_idx','latent_idx'], kind='mergesort')
        .reset_index(drop=True)
    )
    pairs['col'] = np.arange(len(pairs), dtype=np.int64)
    return pairs

def df_to_csr(acts_df: pd.DataFrame,
              feature_index: pd.DataFrame,
              n_examples: Optional[int] = None
) -> csr_matrix:
    """
    Convert a long-form sparse activations DataFrame into a CSR matrix.
    Expects columns: example_idx, instance_idx, latent_idx, activation_value.
    n_examples corresponds to the total number of rows (examples) to 
    allocate in the CSR matrix, it allows for rows with no activations.
    """
    # Merge once to get vectorized 'col' indices
    merged = acts_df.merge(feature_index, on=['instance_idx','latent_idx'], how='left', copy=False)
    rows = merged['example_idx'].to_numpy(dtype=np.int64, copy=False)
    cols = merged['col'].to_numpy(dtype=np.int64, copy=False)
    data = merged['activation_value'].to_numpy(dtype=np.float32, copy=False)
    n_features = int(feature_index['col'].max()) + 1 if len(feature_index) else 0
    if n_examples is None:
        n_examples = int(rows.max()) + 1 if rows.size else 0
    return csr_matrix((data, (rows, cols)), shape=(n_examples, n_features), dtype=np.float32)

def apply_lag_sparse(X_tr: csr_matrix, 
                     X_va: csr_matrix,
                     y_tr: np.ndarray, 
                     y_va: np.ndarray,
                     lag_bins: int):
    """
    Align features/targets with a temporal lag (in bins).
    Positive lag means features lead targets (drop tail of X).
    """
    if lag_bins > 0:
        Xt, yt = X_tr[:-lag_bins], y_tr[lag_bins:]
        Xv, yv = X_va[:-lag_bins], y_va[lag_bins:]
    elif lag_bins < 0:
        k = -lag_bins
        Xt, yt = X_tr[k:], y_tr[:-k]
        Xv, yv = X_va[k:], y_va[:-k]
    else:
        Xt, yt, Xv, yv = X_tr, y_tr, X_va, y_va
    return Xt, Xv, yt, yv

def decode_with_lag_sweep(X_tr: csr_matrix, 
                          X_va: csr_matrix,
                          y_tr: np.ndarray,
                          y_va: np.ndarray,
                          lags=range(0, 6),
                          scaler: Optional[MaxAbsScaler] = None,
                          alpha: float = 30.0,
                          solver: str = 'auto'
) -> dict:
    """
    Ridge decoding on sparse features with a lag sweep.
    Returns dict with best lag, mean R², per-dim R², fitted pipeline, and y_pred.
    """

    best = {'lag': None, 'r2_mean': -np.inf, 'r2_per_dim': None,
            'model': None, 'y_pred': None}

    for lag in lags:
        Xt, Xv, yt, yv = apply_lag_sparse(X_tr, X_va, y_tr, y_va, lag)
        model = make_pipeline(
            scaler,                 # keeps it sparse, scales by max abs
            Ridge(alpha=alpha, solver=solver)
        )
        model.fit(Xt, yt)
        y_pred = model.predict(Xv)
        r2_per_dim = r2_score(yv, y_pred, multioutput='raw_values')
        r2_mean = float(np.mean(r2_per_dim))
        if r2_mean > best['r2_mean']:
            best.update(lag=lag, r2_mean=r2_mean, r2_per_dim=r2_per_dim,
                        model=model, y_pred=y_pred)

    return best
