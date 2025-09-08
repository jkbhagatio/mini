"""Functions for splitting trials into training and validation sets."""

import numpy as np
import pandas as pd

from typing import List, Tuple, Optional

def train_val_split_by_proportion(
    trial_indices: np.ndarray,
    train_proportion: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split unique trials into train/val sets by proportion."""
    unique_trials = pd.Series(trial_indices).dropna().astype(int).unique()

    rng = np.random.default_rng(seed) if shuffle else None
    trials_order = unique_trials.copy()
    if rng is not None:
        rng.shuffle(trials_order)

    n_train = int(len(trials_order) * float(train_proportion))
    train_trials = trials_order[:n_train]
    val_trials = trials_order[n_train:]

    return train_trials, val_trials


def train_val_split_by_session(
    trial_indices: np.ndarray,
    session_ids: np.ndarray,
    train_sessions: List[int],
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split trials into train/val sets by session membership."""
    if not train_sessions:
        raise ValueError("You must provide at least one session index for training.")

    if len(trial_indices) != len(session_ids):
        raise ValueError("`trial_indices` and `session_ids` must be the same length.")

    train_trials = pd.Series(trial_indices[np.isin(session_ids.astype(int), train_sessions)]).dropna().astype(int).unique()
    val_trials   = pd.Series(trial_indices[~np.isin(session_ids.astype(int), train_sessions)]).dropna().astype(int).unique()

    rng = np.random.default_rng(seed) if shuffle else None
    if rng is not None:
        rng.shuffle(train_trials)
        if len(val_trials):
            rng.shuffle(val_trials)

    return train_trials, val_trials