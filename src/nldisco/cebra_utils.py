"""CEBRA-related utilities."""

import numpy as np
import cebra
from scipy.linalg import orthogonal_procrustes

def average_cebra_embeddings_procrustes(model_losses, spikes_train, spikes_val, loss_threshold=3):
    """
    Filter models by loss threshold, align embeddings with Procrustes,
    and return averaged train/val embeddings.
    """
    # Apply loss threshold
    model_keep = [(p, l) for p, l in model_losses if l <= loss_threshold]
    print(f"Using {len(model_keep)} models after applying loss threshold {loss_threshold}.")

    # Get embeddings 
    emb_train_list, emb_val_list = [], []
    for p, _ in model_keep:
        model = cebra.CEBRA.load(p, weights_only=False)
        emb_train_list.append(model.transform(spikes_train).astype(np.float64))
        emb_val_list.append(model.transform(spikes_val).astype(np.float64))

    # Procrustes helpers
    def center(X, mu): return X - mu
    def fit_alignment_to_ref(ref_tr, X_tr, allow_scaling=False):
        mu_ref, mu_X = ref_tr.mean(axis=0, keepdims=True), X_tr.mean(axis=0, keepdims=True)
        A, B = center(X_tr, mu_X), center(ref_tr, mu_ref)
        R, s = orthogonal_procrustes(A, B)
        if not allow_scaling: s = 1.0
        return R, s, mu_X
    def apply_alignment(X, R, s, mu_src): return center(X, mu_src) @ R * s

    # reference = first model's train embedding
    ref_tr = emb_train_list[0]

    aligned_train, aligned_val = [], []
    for Etr, Eva in zip(emb_train_list, emb_val_list):
        R, s, mu_src = fit_alignment_to_ref(ref_tr, Etr)
        aligned_train.append(apply_alignment(Etr, R, s, mu_src))
        aligned_val.append(apply_alignment(Eva, R, s, mu_src))

    # Average across models, per split
    avg_train = np.mean(np.stack(aligned_train, axis=0), axis=0)
    avg_val   = np.mean(np.stack(aligned_val, axis=0), axis=0)

    # Final recentering
    avg_train -= avg_train.mean(axis=0, keepdims=True)
    avg_val   -= avg_val.mean(axis=0, keepdims=True)

    print("Train averaged embedding shape:", avg_train.shape)
    print("Val averaged embedding shape:", avg_val.shape)
    return avg_train, avg_val
