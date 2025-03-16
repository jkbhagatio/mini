"""Model training."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch as t
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int
from torch import bfloat16, nn, Tensor
from torch.nn import functional as F
from tqdm.notebook import tqdm

# <s> SAE class config

@dataclass
class SaeConfig:
    """Config class to set some params for SAE."""
    n_input_ae: int  # number of input units to the autoencoder
    n_hidden_ae: int  # number of h_ae units in the autoencoder
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
        frac_active_thresh: float = 1e-3,  # fraction of time a neuron needs to be active to be alive
        resample_thresh: float = 0.1  # threshold of fraction of dead neurons, above which we resample
    ) -> float:
        """Resamples dead neurons according to `frac_active`, returns frac_dead."""
        # Get a tensor of dead neurons.
        dead_features_mask = frac_active < frac_active_thresh
        n_dead = dead_features_mask.sum().item()
        frac_dead = n_dead / dead_features_mask.numel()
        print(f"{frac_dead=}", end="")

        if (frac_dead < resample_thresh) or (step > final_resample_step):
            print()
            return

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

def lmse(
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
    model: Sae,
    optimizer: t.optim.Optimizer,
    spike_counts: Int[Tensor, "n_examples n_units"],
    batch_sz: int,
    n_steps: int,
    log_freq: int,
    seq_len: int,  # number of timebins to use in each spike_count_seq
    use_lr_sched: bool = False,
    neuron_resample_window: Optional[int] = None,  # in number of steps
):
    """Optimizes the autoencoder using the given hyperparameters."""
    # Create lists to store data we'll eventually be plotting.
    frac_active_all_steps = []  # fraction of non-zero activations for each neuron (feature)
    data_log = {
        "frac_active": {},
        "loss": {},
        "l0": {}
    }

    # Define valid samples for `spike_counts`.
    n_examples, _n_units = spike_counts.shape
    valid_starts = n_examples - seq_len + 1

    if use_lr_sched:
        init_lr = optimizer.param_groups[0]["lr"]
        min_lr = init_lr * 1e-2

    pbar = tqdm(range(n_steps))
    for step in pbar:
        # Check for dead neurons and resample them if found.
        if (neuron_resample_window is not None)  and ((step + 1) % neuron_resample_window == 0):
            frac_active_in_window = reduce(
                t.stack(frac_active_all_steps[-neuron_resample_window:], dim=0),
                "window inst hidden_ae -> inst hidden_ae", 
                "mean"
            )
            data_log["frac_active"][step] = frac_active_in_window.detach().cpu()
            model.resample_neurons(
                frac_active_in_window, 
                step, 
                final_resample_step=(n_steps // 2),
                frac_active_thresh=(1 / model.cfg.n_hidden_ae)
            )

        if use_lr_sched:
            optimizer.param_groups[0]["lr"] = (
                simple_cosine_lr_sched(step, n_steps, init_lr, min_lr)
            )

        # Get batch of spike counts to feed into SAE.
        start_idxs = t.randint(0, valid_starts, (batch_sz, model.cfg.n_instances))
        seq_idxs = start_idxs.unsqueeze(-1) + t.arange(seq_len)  # broadcast seq idxs to new dim
        spike_count_seqs = spike_counts[seq_idxs]  # [batch_sz, n_instances, seq_len, n_units]

        # Optimize.
        optimizer.zero_grad()
        spike_count_recon, h = model(spike_count_seqs)

        # Example lmse loss, no l1 loss.
        # take loss between reconstructions and last timebin (sequence) of spike_count_seqs.
        loss = lmse(spike_count_seqs[..., -1, :], spike_count_recon, tau=1)
        loss = reduce(loss, "batch inst -> ", "mean")

        # Example mse loss, no l1 loss.
        # loss = mse(spike_count_seqs, spike_count_recon)
        # loss = reduce(loss, "batch inst -> ", "mean")
        
        # Example mse loss with vanilla-l1-loss.
        # l1_loss_val = l1_loss(z, lamda=5e-4)
        # l2_loss_val = mse(spike_count_seqs, spike_count_recon)
        # loss = reduce(l1_loss_val + l2_loss_val, "batch inst -> ", "mean")
        # model.normalize_decoder()

        # Example mse loss with decoder-norm-l1-loss.
        # l1_loss_val = l1_loss_decoder_norm(z, model.W_dec, lamda=5e-4)
        # l2_loss_val = mse(spike_count_seqs, spike_count_recon)
        # loss = reduce(l1_loss_val + l2_loss_val, "batch inst -> ", "mean")

        # Example mse loss with tanh-l1-loss.
        # l1_loss_val = tanh_loss(z, model.W_dec, lamda=5e-4, A=1.0, B=0.2)
        # l2_loss_val = mse(spike_count_seqs, spike_count_recon)
        # loss = reduce(l1_loss_val + l2_loss_val, "batch inst -> ", "mean")
        
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
            data_log["l0"][step] = l0
            data_log["loss"][step] = loss.item()

    return data_log

# </s>

# <s> Other

def batched_forward_pass(
    spike_counts: Int[Tensor, "n_examples n_units"],
    sae: Sae, 
    batch_sz: int, 
    seq_len: int, 
    device: str = "cuda"
) -> tuple[
    Int[Tensor, "n_examples n_inst"], 
    Float[Tensor, "n_examples n_inst n_units"], 
    Float[Tensor, "n_examples n_inst n_hidden_ae"],
    Float[Tensor, "n_examples n_inst"],
    Float[Tensor, "n_examples n_inst"],
    Float[Tensor, "n_examples n_inst"]
]:
    """Perform batched forward pass through SAE model.
    
    Returns a tuple containing:
        - L0 (number of active features per example)
        - Reconstructed spike counts per example
        - Hidden layer activations per example
        - L1 loss per example
        - L2 loss per example
        - Total loss per example
    """
    # Initialize batches.
    n_examples, n_units = spike_counts.shape[0], spike_counts.shape[1]
    n_inst = sae.cfg.n_instances
    n_full_batches, final_batch_sz = n_examples // batch_sz, n_examples % batch_sz
    n_steps = n_full_batches + (1 if final_batch_sz > 0 else 0)
    
    # Initialize output tensors.
    l0 = t.zeros((n_examples, n_inst), dtype=t.float32, device=device)
    recon_spk_cts = t.empty((n_examples, n_inst, n_units), dtype=t.bfloat16, device=device)
    h_acts = t.empty((n_examples, n_inst, sae.cfg.n_hidden_ae), dtype=t.bfloat16, device=device)
    l1_losses = t.empty((n_examples, n_inst), dtype=t.bfloat16, device=device)
    l2_losses = t.empty((n_examples, n_inst), dtype=t.bfloat16, device=device)
    total_losses = t.empty((n_examples, n_inst), dtype=t.bfloat16, device=device)
    
    progress_bar = tqdm(range(n_steps))
    with t.no_grad():
        for step in progress_bar:
            # Set up for forward pass.
            cur_batch_size = batch_sz if step < n_full_batches else final_batch_sz
            start_idx = step * batch_sz
            end_idx = start_idx + cur_batch_size
            idxs = t.arange(start_idx, end_idx)
            idxs = repeat(idxs, "batch -> batch inst", inst=n_inst)
            # broadcast idxs for each sequence to a new dimension
            seq_idxs = idxs.unsqueeze(-1) + t.arange(seq_len)
            spike_count_seqs = spike_counts[seq_idxs]
            spike_count_seqs = rearrange(
                spike_count_seqs, "batch inst seq unit -> (batch seq) inst unit"
            )
            
            # Compute reconstructions, hidden layer activations, and losses.
            l1_loss, l2_loss, loss, z, x_prime = sae(spike_count_seqs)
            
            # Compute L0.
            nonzero_mask = (z.abs() > 1e-7).float()
            cur_l0 = reduce(nonzero_mask, "batch inst sae_feat -> batch inst", "sum")
            
            # Store results
            l0[idxs[:, 0]] = cur_l0
            recon_spk_cts[idxs[:, 0]] = x_prime
            h_acts[idxs[:, 0]] = z
            l1_losses[idxs[:, 0]] = l1_loss
            l2_losses[idxs[:, 0]] = l2_loss
            total_losses[idxs[:, 0]] = loss
    
    return l0, recon_spk_cts, h_acts, l1_losses, l2_losses, total_losses

# </s>
