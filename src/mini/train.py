"""Model training."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch as t
import wandb
from einops import asnumpy, einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from torch import bfloat16, nn, Tensor
from torch.nn import functional as F
from tqdm import tqdm

from mini import plot as mp
from mini.util import vec_r2

# <s> SAE class config

@dataclass
class SaeConfig:
    """Config class to set some params for SAE."""
    n_input_ae: int  # number of input units to the autoencoder
    n_hidden_ae: int  # number of hidden units in the autoencoder
    seq_len: int = 1  # number of time bins in a sequence
    n_instances: int = 2  # number of model instances to optimize in parallel
    topk: Optional[int] = None  # avg num of contributing sae features per example in a batch


class Sae(nn.Module):
    """SAE model for learning sparse representations of binned spike counts."""
    # Shapes of weights and biases for the encoder and decoder in the single-layer SAE.
    W_enc: Float[Tensor, "inst in_ae h_ae"]
    W_dec: Float[Tensor, "inst h_ae in_ae"]
    b_enc: Float[Tensor, "inst h_ae"]
    b_dec: Float[Tensor, "inst in_ae"]

    def __init__(self, cfg: SaeConfig):
        """Initializes model parameters."""
        super().__init__()
        self.cfg = cfg
        
        # Tied weights initialization to reduce dead neurons (https://arxiv.org/pdf/2406.04093).
        in_dim = cfg.n_input_ae * cfg.seq_len  # expand input dim for sequences
        self.W_enc = t.empty((cfg.n_instances, cfg.n_hidden_ae, in_dim), dtype=bfloat16)
        self.W_enc = nn.init.kaiming_normal_(self.W_enc, mode="fan_in", nonlinearity="relu")
        self.W_dec = rearrange(
            self.W_enc[..., :cfg.n_input_ae], "inst h_ae in_ae -> inst in_ae h_ae"
        ).clone()
        self.W_enc, self.W_dec = nn.Parameter(self.W_enc), nn.Parameter(self.W_dec)
        
        self.b_enc = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_hidden_ae), dtype=bfloat16))
        self.b_dec = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_input_ae), dtype=bfloat16))

    def forward(self, x: Float[Tensor, "batch inst seq in_ae"]) -> (
        Tuple[Float[Tensor, "batch inst h_ae"], Float[Tensor, "batch inst in_ae"]]
    ):
        """Computes loss as a function of SAE feature sparsity and spike_count reconstructions."""
        # Compute encoder h_ae activations.
        x = rearrange(x, "batch inst seq in_ae -> batch inst (seq in_ae)")
        h = einsum(x, self.W_enc, "batch inst in_dim, inst h_ae in_dim -> batch inst h_ae")
        h = F.relu(h + self.b_enc)
        
        if self.cfg.topk:
            # Batch topk: only keep the (top k features * batch_sz) in the batch.
            batch_topk = h.shape[0] * self.cfg.n_instances * self.cfg.topk
            feat_keep_vals, feat_keep_idxs = h.ravel().topk(batch_topk)
            h = h.ravel().zero_().scatter_(0, feat_keep_idxs, feat_keep_vals).view_as(h)

        # Compute reconstructed input.
        x_prime = (
            einsum(h, self.W_dec, "batch inst h_ae, inst in_ae h_ae -> batch inst in_ae")
            + self.b_dec
        )
        x_prime = F.relu(x_prime)  # ensure reconstructed spikes are non-negative

        return x_prime, h

    @t.no_grad()
    def resample_neurons(
        self, 
        frac_active: Float[Tensor, "inst hidden_ae"],
        step: int,
        final_resample_step: int,
        # fraction of examples a neuron needs to be active to be alive
        frac_active_thresh: float = 1e-3,
        # threshold of fraction of dead neurons, above which we resample
        resample_thresh: float = 0.1
    ) -> float:
        """Resamples dead neurons according to `frac_active`, returns the fraction dead."""
        # Get a tensor of dead neurons.
        dead_features_mask = frac_active < frac_active_thresh
        n_dead = dead_features_mask.sum().item()
        frac_dead = n_dead / dead_features_mask.numel()
        print(f"{frac_dead=}", end="")

        if (frac_dead < resample_thresh) or (step > final_resample_step):
            print()
            return frac_dead

        print(";  Resampling neurons.")
        
        # Create new weights
        replacements = t.randn(
            (n_dead, self.cfg.n_input_ae * self.cfg.seq_len), 
            device=self.W_enc.device, 
            dtype=bfloat16
        )
        # normalize to match existing weight scale
        current_weight_scale = self.W_enc[~dead_features_mask].norm(dim=-1).mean()
        replacements_norm = (
            replacements * (current_weight_scale / (replacements.norm(dim=-1, keepdim=True) + 1e-6))
        )
        
        # Update weights
        new_W_enc = self.W_enc.clone()
        new_W_enc[dead_features_mask] = replacements_norm
        self.W_enc.copy_(new_W_enc)
        self.W_dec.copy_(
            rearrange(self.W_enc[..., :self.cfg.n_input_ae], "inst h_ae in_ae -> inst in_ae h_ae")
        )
        self.b_enc[dead_features_mask].fill_(0.0)

        return frac_dead
    
    @t.no_grad()
    def normalize_decoder(self):
        """Unit norms the decoder weights."""
        self.W_dec.data = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-6)

# </s>

# <s> Loss and optimization functions

# <ss> Loss functions for reconstruction.

def mse(
    x: Float[Tensor, "batch inst in_ae"],  # input
    x_prime: Float[Tensor, "batch inst in_ae"],  # reconstruction
) -> Float[Tensor, "batch inst in_ae"]:
    """Computes the mean squared error loss between true input and reconstruction."""
    return reduce((x - x_prime).pow(2), "batch inst in_ae -> batch inst", "mean")

def msle(
    x: Float[Tensor, "batch inst in_ae"],  # input
    x_prime: Float[Tensor, "batch inst in_ae"],  # reconstruction
    tau: int = 1,  # relative overestimation/underestimation penalty (1 for symmetric)
) -> Float[Tensor, "batch inst in_ae"]:
    """Computes the log mean squared error loss between true input and reconstruction."""
    return reduce(
        (tau * t.log(x_prime + 1) - t.log(x + 1)).pow(2), "batch inst in_ae -> batch inst", "mean"
    )

# </ss>

# <ss> Loss functions for sparsity.

def l1_loss(
    z: Float[Tensor, "batch inst hidden_ae"],  # hidden activations
    lamda: float = 1e-5,  # sparsity penalty coefficient
) -> Float[Tensor, "batch inst"]:
    """Computes sparsity penalty based on the l1 norm of the activations."""
    return lamda * reduce(z.abs(), "batch inst hidden_ae -> batch inst", "sum")


def l1_loss_decoder_norm(
    z: Float[Tensor, "batch inst hidden_ae"],  # hidden activations
    W_dec: Float[Tensor, "inst in_ae hidden_ae"],  # decoder weights
    lamda: float = 1e-5,   # sparsity penalty coefficient
) -> Float[Tensor, "batch inst"]:
    """Computes sparsity penalty based on the norm of a feature's decoder weights. 
    
    Where each of these feature norms is modulated by the l1 norm of the respective activation.
    See https://transformer-circuits.pub/2024/april-update/index.html#training-saes for details.
    """
    sparsity_loss = einsum(
        z.abs(), t.norm(W_dec, p=2, dim=1), "batch inst hidden_ae, inst hidden_ae -> batch inst"
    )
    return lamda * sparsity_loss


def tanh_loss(
    z: Float[Tensor, "batch inst hidden_ae"],  # hidden activations
    W_dec: Float[Tensor, "inst in_ae hidden_ae"],  # decoder weights
    lamda: float = 1.0,  # sparsity penalty coefficient
    A: float = 1.0,  # tanh sparsity penalty scaling factor
    B: float = 1.0  # tanh saturation scaling factor
 ) -> Float[Tensor, "batch inst"]:
    """Computes sparsity penalty using tanh to combat shrinkage of meaningful activations.

    See https://transformer-circuits.pub/2024/feb-update/index.html#dict-learning-tanh for details.
    """
    sparsity_loss = l1_loss_decoder_norm(z, W_dec, lamda)
    return (A / B) * t.tanh(B * sparsity_loss)

# </ss>

def simple_cosine_lr_sched(step: int, n_steps: int, initial_lr: float, min_lr: float):
    """Learning rate schedule with warmup, decay and cosyne cycle."""
    n_warmup_steps = int(n_steps * 0.1)
    decay_start_step = int(n_steps * 0.5)
    n_decay_steps = n_steps - decay_start_step
    n_cycle_steps = int(n_decay_steps * 0.2)
    
    # Warmup phase
    if step < n_warmup_steps:
        return max(initial_lr * (step / n_warmup_steps), min_lr)
    
    # Decay phase: cosine decay with cycles
    if step >= decay_start_step:
        decay_steps = n_steps - decay_start_step
        decay_position = (step - decay_start_step) / decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_position))
        decayed_lr = min_lr + (initial_lr - min_lr) * cosine_decay
        cycle_position = ((step - decay_start_step) % n_cycle_steps) / n_cycle_steps
        cycle_factor = 0.5 * (1 + math.cos(2 * math.pi * cycle_position))
        cycle_amplitude = 0.1 * (initial_lr - min_lr)
            
        return decayed_lr + cycle_amplitude * cycle_factor

    # Constant phase: between warmup and decay start
    return initial_lr


def optimize(
    spk_cts: Int[Tensor, "n_examples n_units"],
    model: Sae,
    seq_len: int,  # number of timebins to use in each spike_count_seq
    loss_fn: str,
    lr: float,
    use_lr_sched: bool,
    neuron_resample_window: int,  # in number of steps
    batch_sz: int,
    n_steps: int,
    log_freq: int,
    log_wandb: bool = False,
    plot_l0: bool = False
):
    """Optimizes the autoencoder using the given hyperparameters."""
    # Create lists to store data we"ll eventually be plotting.
    frac_active_all_steps = []  # fraction of non-zero activations for each neuron (feature)
    l0_history = []  # history of l0 mean and std for each step
    data_log = {
        "frac_active": {},
        "loss": {},
        "l0": {}
    }
    frac_dead = None

    # Define valid samples for `spk_cts`.
    n_examples, _n_units = spk_cts.shape
    valid_starts = n_examples - seq_len + 1

    # Define the optimizer.
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    if use_lr_sched:
        min_lr = lr * 1e-2

    # Set `tau` for msle loss if needed.
    tau = float(loss_fn.split("_")[-1]) if "msle" in loss_fn else 0.0

    # Loop over the data.
    pbar = tqdm(range(n_steps), desc="SAE batch training step")
    for step in pbar:
        
        # Check for dead neurons and resample them if found.
        if (neuron_resample_window is not None) and ((step + 1) % neuron_resample_window == 0):
            frac_active_in_window = reduce(
                t.stack(frac_active_all_steps[-neuron_resample_window:], dim=0),
                "window inst hidden_ae -> inst hidden_ae", 
                "mean"
            )
            data_log["frac_active"][step] = frac_active_in_window.detach().cpu()
            frac_dead = model.resample_neurons(
                frac_active_in_window, 
                step, 
                final_resample_step=(n_steps // 2),
                frac_active_thresh=5e-5
                # frac_active_thresh=(1 / model.cfg.n_hidden_ae)
            )

        # Update lr.
        if use_lr_sched:
            optimizer.param_groups[0]["lr"] = (
                simple_cosine_lr_sched(step, n_steps, lr, min_lr)
            )

        # Get batch of spike counts to feed into SAE.
        start_idxs = t.randint(0, valid_starts, (batch_sz, model.cfg.n_instances))
        seq_idxs = start_idxs.unsqueeze(-1) + t.arange(seq_len)  # broadcast seq idxs to new dim
        spike_count_seqs = spk_cts[seq_idxs]  # [batch_sz, n_instances, seq_len, n_units]

        # Optimize.
        optimizer.zero_grad()
        spike_count_recon, h = model(spike_count_seqs)
        # take loss between reconstructions and last timebin (sequence) of spike_count_seqs
        if loss_fn == "mse":
            loss = mse(spike_count_seqs[..., -1, :], spike_count_recon)
        elif "msle" in loss_fn:
            loss = msle(spike_count_seqs[..., -1, :], spike_count_recon, tau=tau)
        else:
            raise ValueError(f"Invalid loss function: {loss_fn}")
        loss = reduce(loss, "batch inst -> ", "mean")   
        loss.backward()
        optimizer.step()

        # Calculate the sparsities and add them to the list.
        frac_active_batch = reduce(
            (h.abs() > 1e-6).float(), "batch inst hidden_ae -> inst hidden_ae", "mean"
        )
        frac_active_all_steps.append(frac_active_batch)

        # Display progress bar, and append new values for plotting.
        if step % log_freq == 0 or (step + 1 == n_steps):
            l0 = reduce((h.abs() > 1e-6).float(), "batch inst hidden_ae -> batch inst", "sum")
            l0_mean, l0_std = l0.mean().item(), l0.std().item()
            pbar.set_postfix(loss=f"{loss.item():.5f},  {l0_mean=}, {l0_std=}")
            data_log["l0"][step] = {"mean": l0_mean, "std": l0_std}
            data_log["loss"][step] = loss.item()

            if log_wandb:
                wandb.log({"loss": loss.item(), "l0_mean": l0_mean, "l0_std": l0_std, "step": step})
            
                if frac_dead:
                    wandb.log({"frac_dead": frac_dead, "step": step})
            
                if plot_l0:
                    alpha = 0.3 + (0.7 * step / n_steps)  # alpha from 0.3 to 1.0
                    l0_history.append({"step": step, "mean": l0_mean, "std": l0_std, "alpha": alpha})
                    l0_fig = mp.plot_l0_stats(l0_history)
                    wandb.log({"l0_std_vs_mean": l0_fig, "step": step})

    return data_log


def eval_model(
    spk_cts: Int[Tensor, "n_examples n_units"],
    sae: Sae,
    batch_sz: int = 1024,
    log_wandb: bool = False
):
    """Evaluates the model after training, and generates plots/metrics.
    
    Plots/Metrics:
    1. L0 boxplot (per example)
    2a. Cosine-Similarity boxplot of reconstructions vs. true over all neurons (per example)
    2b. Cosine-Similarity boxplot of reconstructions vs. true over all examples (per neuron)
    3b. R² boxplot of reconstructions vs. true over all neurons (per example)
    3a. R² boxplot of reconstructions vs. true over all examples (per neuron)

    """
    device = sae.W_enc.device

    # Set some constants.
    n_inst = sae.cfg.n_instances
    n_units = spk_cts.shape[1]
    n_examples = spk_cts.shape[0]
    valid_starts = n_examples - sae.cfg.seq_len + 1
    n_steps = valid_starts // batch_sz  # total number of examples
    n_recon_examples = n_steps * batch_sz
    
    # <ss> Run examples through model and compute metrics.

    # Create tensors to store L0 and reconstructions.
    l0 = t.zeros((n_recon_examples, n_inst), dtype=t.float32, device=device)
    recon_spk_cts = t.empty((n_recon_examples, n_inst, n_units), dtype=bfloat16, device=device)

    # Create tensors to store eval metrics.
    r2_per_example = t.empty((n_recon_examples, n_inst), dtype=bfloat16, device=device)
    cos_sim_per_example = t.empty((n_recon_examples, n_inst), dtype=bfloat16, device=device)

    progress_bar = tqdm(range(n_steps), desc="SAE batch evaluation step")
    with t.no_grad():
        for step in progress_bar:  # loop over all examples
            # Get start index for each seq in batch, and then get the full seq indices.
            start_idxs = t.arange(step * batch_sz, (step + 1) * batch_sz)
            seq_idxs = repeat(start_idxs, "batch -> batch inst", inst=n_inst)
            seq_idxs = seq_idxs.unsqueeze(-1) + t.arange(sae.cfg.seq_len)  # broadcast to seq dim
            spike_count_seqs = spk_cts[seq_idxs]  # [batch, inst, seq, unit]
            # Forward pass through SAE.
            x_prime, h = sae(spike_count_seqs)
            nonzero_mask = (h.abs() > 1e-7).float()
            cur_l0 = reduce(nonzero_mask, "batch inst sae_feat -> batch inst", "sum")
            # Store results.
            l0[start_idxs] = cur_l0
            recon_spk_cts[start_idxs] = x_prime
            # Calculate metrics for examples.
            r2_per_example[start_idxs] = vec_r2(x_prime, spk_cts[start_idxs])
            cos_sim_per_example[start_idxs] = (
                t.cosine_similarity(x_prime, spk_cts[start_idxs].unsqueeze(1), dim=-1)
            )

    r2_per_example[~t.isfinite(r2_per_example)] = 0.0  # div by 0 cases

    # Calculate metrics for units.
    cos_sim_per_unit = t.empty((n_units, n_inst))
    r2_per_unit = np.empty((n_units, n_inst))

    spk_cts_np = asnumpy(spk_cts.float())
    recon_spk_cts_np = asnumpy(recon_spk_cts.float())

    for unit in range(n_units):
        cos_sim_per_unit[unit] = t.cosine_similarity(
            recon_spk_cts[..., unit], spk_cts[:n_recon_examples, unit].unsqueeze(-1), dim=0
        )
        for inst in range(n_inst):
            r2_per_unit[unit, inst] = r2_score(
                spk_cts_np[:n_recon_examples, unit], recon_spk_cts_np[:, inst, unit]
        )

    # </ss>

    # <ss> Create plots.

    fig, (ax_l0, ax_r2, ax_cos) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    sns.set_theme(style="whitegrid")
    
    # <sss> L0 boxplot.
    
    l0_data = [asnumpy(l0[:, i]) for i in range(n_inst)]
    mp.box_strip_plot(
        ax=ax_l0,
        data=l0_data,
        show_legend=True
    )
    # Update width of box plot and size and alpha of strip plot.
    # for box in ax_l0.artists:
    #     box.set_width(0.4)  # Change boxplot width
    # for collection in ax_l0.collections:
    #     if isinstance(collection, plt.matplotlib.collections.PathCollection):
    #         collection.set_sizes([4])
    #         collection.set_alpha(0.4)
    # Prettify axes.
    ax_l0.set_xlabel("")
    ax_l0.set_ylabel("")
    ax_l0.set_xticks(range(n_inst))
    ax_l0.set_xticklabels([f"SAE {i}" for i in range(n_inst)])
    ax_l0.set_yticks(np.arange(0, l0.max().item() + 1, sae.cfg.topk // 2))
    ax_l0.set_title("L0 of SAE features")
    
    # </sss>

    # <sss> Format and plot R² and Cosine Similarity data.
    
    cos_sim_per_example = asnumpy(cos_sim_per_example.float())
    r2_per_example = asnumpy(r2_per_example.float())
    cos_sim_per_unit = asnumpy(cos_sim_per_unit.float())

    model_names = [f"SAE {i}" for i in range(2)]
    dfs = []

    dfs.append(
        pd.DataFrame(cos_sim_per_example, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
        .assign(Type="Examples", Metric="Cosine Similarity")
    )

    dfs.append(
        pd.DataFrame(cos_sim_per_unit, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
        .assign(Type="Units", Metric="Cosine Similarity")
    )

    dfs.append(
        pd.DataFrame(r2_per_example, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
        .assign(Type="Examples", Metric="R²")
    )

    dfs.append(
        pd.DataFrame(r2_per_unit, columns=model_names)
        .melt(var_name="SAE", value_name="Value")
        .assign(Type="Units", Metric="R²")
    )

    df = pd.concat(dfs, ignore_index=True)

    cos_sim_df = df[df["Metric"] == "Cosine Similarity"]
    r2_df = df[df["Metric"] == "R²"]
    
    mp.box_strip_plot(
        ax=ax_r2,
        data=r2_df,
        x="Type",
        y="Value",
        hue="SAE",
        show_legend=False
    )
    # Prettify axes.
    ax_r2.set_xlabel("")
    ax_r2.set_ylabel("")
    ax_r2.set_ylim(-1.0, 1.0)
    ax_r2.set_yticks(np.arange(-1.0, 1.1, 0.1))
    ax_r2.set_title("R² of SAE reconstructions")

    mp.box_strip_plot(
        ax=ax_cos,
        data=cos_sim_df,
        x="Type",
        y="Value",
        hue="SAE",
        show_legend=False
    )
    # Prettify axes.
    ax_cos.set_xlabel("")
    ax_cos.set_ylabel("")
    ax_cos.set_ylim(0.1, 1.0)
    ax_cos.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax_cos.set_title("Cosine Similarity of true and reconstructed spike counts")
    
    # </sss>

    # </ss>

    # <ss> Log to wandb.

    if log_wandb:

        # Log metrics figure
        wandb.log({"combined_metrics_plot": wandb.Image(fig)})
        plt.close(fig)
        
        # Log metrics values
        wandb.log({
            "r2_per_example_mean": np.mean(r2_per_example),
            "r2_per_unit_mean": np.mean(r2_per_unit),
            "cos_per_example_mean": np.mean(cos_sim_per_example),
            "cos_per_unit_mean": np.mean(cos_sim_per_unit),
        })

    # </ss>

# </s>
