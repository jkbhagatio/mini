"""Test wandb sweeps."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import submitit
import torch as t
import wandb
import yaml
from einops import asnumpy, einsum, rearrange, reduce, repeat, pack, parse_shape, unpack
from jaxtyping import Float, Int, BFloat16
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch import bfloat16, nn, Tensor
from tqdm.notebook import tqdm

from mini import train as mt


def optimize(
    spk_cts: Int[Tensor, "n_examples n_units"],
    model: mt.Sae,
    seq_len: int,  # number of timebins to use in each spike_count_seq
    loss_fn: Callable,
    lmse_tau: float,
    lr: float,
    use_lr_sched: bool,
    neuron_resample_window: int,  # in number of steps
    batch_sz: int,
    n_steps: int,
    log_freq: int,
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

    # Define valid samples for `spk_cts`.
    n_examples, _n_units = spk_cts.shape
    valid_starts = n_examples - seq_len + 1

    # Define the optimizer.
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    if use_lr_sched:
        min_lr = lr * 1e-2

    pbar = tqdm(range(n_steps))
    for step in pbar:
        
        # Check for dead neurons and resample them if found.
        frac_dead = None
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
                frac_active_thresh=(1 / model.cfg.n_hidden_ae)
            )

        # Update lr.
        if use_lr_sched:
            optimizer.param_groups[0]["lr"] = (
                mt.simple_cosine_lr_sched(step, n_steps, lr, min_lr)
            )

        # Get batch of spike counts to feed into SAE.
        start_idxs = t.randint(0, valid_starts, (batch_sz, model.cfg.n_instances))
        seq_idxs = start_idxs.unsqueeze(-1) + t.arange(seq_len)  # broadcast seq idxs to new dim
        spike_count_seqs = spk_cts[seq_idxs]  # [batch_sz, n_instances, seq_len, n_units]

        # Optimize.
        optimizer.zero_grad()
        spike_count_recon, h = model(spike_count_seqs)
        # take loss between reconstructions and last timebin (sequence) of spike_count_seqs
        if loss_fn == mt.mse:
            loss = loss_fn(spike_count_seqs[..., -1, :], spike_count_recon)
        elif loss_fn == mt.lmse:
            loss = loss_fn(spike_count_seqs[..., -1, :], spike_count_recon, tau=lmse_tau)
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
            wandb.log({"loss": loss.item(), "l0_mean": l0_mean, "l0_std": l0_std, "step": step})
            
            # Create and log l0 std vs mean plot with alpha based on step progress
            alpha = 0.3 + (0.7 * step / n_steps)  # Alpha from 0.3 to 1.0
            l0_history.append(
                {
                    "step": step,
                    "mean": l0_mean,
                    "std": l0_std,
                    "alpha": alpha
                }
            )
            l0_fig = go.Figure()
            for point in l0_history:
                l0_fig.add_trace(
                    go.Scatter(
                        x=[point["mean"]],
                        y=[point["std"]],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=f"rgba(0, 0, 255, {point['alpha']})"
                        ),
                        name=f"Step {point['step']}",
                        showlegend=False  # Optional: disable legend to avoid clutter
                    )
                )
            l0_fig.update_layout(
                title="L0 std vs mean",
                xaxis_title="L0 mean",
                yaxis_title="L0 std"
            )
            wandb.log({"l0_std_vs_mean": l0_fig, "step": step})
            if frac_dead is not None:
                wandb.log({"frac_dead": frac_dead, "step": step})

    return data_log


def vec_r2(y_pred: Float[Tensor, "n inst feat"], y_true: Float[Tensor, "n feat"]) -> (
    Float[Tensor, "n"]
):
    """Calculates vectorized R² scores for each example in a batch."""
    # Calculate SST
    y_true_mean = reduce(y_true, "n feat -> n", "mean")
    ss_tot = reduce((y_true - y_true_mean.unsqueeze(-1)) ** 2, "n feat -> n", "sum")
    # Calculate SSR
    y_true = repeat(y_true, "n feat -> n inst feat", inst=y_pred.shape[1])  # match to broadcast
    ss_res = reduce((y_true - y_pred) ** 2, "n inst feat -> n inst", "sum")
    
    return 1 - (ss_res / ss_tot.unsqueeze(-1))


def eval_model(
    spk_cts: Int[Tensor, "n_examples n_units"],
    sae: mt.Sae,
    batch_sz: int = 1024
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
    
    # <s> Run examples through model and compute metrics.

    # Create tensors to store L0 and reconstructions.
    l0 = t.zeros((n_recon_examples, n_inst), dtype=t.float32, device=device)
    recon_spk_cts = t.empty((n_recon_examples, n_inst, n_units), dtype=bfloat16, device=device)

    # Create tensors to store eval metrics.
    r2_per_example = t.empty((n_recon_examples, n_inst), dtype=bfloat16, device=device)
    cos_sim_per_example = t.empty((n_recon_examples, n_inst), dtype=bfloat16, device=device)

    progress_bar = tqdm(range(n_steps))
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

    pbar_unit = tqdm(range(n_units))
    for unit in pbar_unit:
        cos_sim_per_unit[unit] = t.cosine_similarity(
            recon_spk_cts[..., unit], spk_cts[:n_recon_examples, unit].unsqueeze(-1), dim=0
        )
        for inst in range(n_inst):
            r2_per_unit[unit, inst] = r2_score(
                spk_cts_np[:n_recon_examples, unit], recon_spk_cts_np[:, inst, unit]
        )

    # </s>

    # <s> Create plots.

    fig, (ax_l0, ax_r2, ax_cos) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    sns.set(style="whitegrid")
    
    # <ss> L0 boxplot.
    
    l0_data = [asnumpy(l0[:, i]) for i in range(n_inst)]
    sns.boxplot(
        data=l0_data,
        width=0.4,
        showfliers=False,
        showmeans=True,
        meanprops={"markersize": "7", "markerfacecolor": "white", "markeredgecolor": "white"},
        ax=ax_l0,
    )
    sns.stripplot(
        data=l0_data,
        size=2,
        alpha=0.4,
        jitter=True,
        dodge=True,
        ax=ax_l0,
    )
    ax_l0.set_xlabel("")
    ax_l0.set_ylabel("")
    ax_l0.set_title("L0 of SAE features")
    ax_l0.set_xticklabels([f"SAE {i}" for i in range(n_inst)])
    ax_l0.set_yticks(np.arange(0, l0.max().item() + 1, sae.cfg.topk // 2))
    
    # </ss>

    # <ss> Format R² and Cosine Similarity data for plotting.
    
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
    
    # </ss>

    # <ss> Add R² and Cosine Similarity boxplots to the figure.

    sns.boxplot(
        data=r2_df,
        x="Type", 
        y="Value", 
        hue="SAE",
        width=0.4,
        showfliers=False,
        showmeans=True,
        meanprops={"markersize": "7", "markerfacecolor": "white", "markeredgecolor": "white"},
        legend=False,
        ax=ax_r2,
    )
    sns.stripplot(
        data=r2_df,
        x="Type",
        y="Value",
        hue="SAE",
        size=2,
        alpha=0.4,
        dodge=True, 
        jitter=True,
        legend=False,
        ax=ax_r2,
    )

    ax_r2.set_title("R² of SAE reconstructions")
    ax_r2.set_ylabel("")
    ax_r2.set_xlabel("")
    ax_r2.set_ylim(-1.0, 1.0)
    ax_r2.set_yticks(np.arange(-1.0, 1.1, 0.2))

    sns.boxplot(
        data=cos_sim_df,
        x="Type", 
        y="Value", 
        hue="SAE",
        width=0.4,
        showfliers=False,
        showmeans=True,
        meanprops={"markersize": "7", "markerfacecolor": "white", "markeredgecolor": "white"},
        legend=False,
        ax=ax_cos,
    )
    sns.stripplot(
        data=cos_sim_df,
        x="Type",
        y="Value",
        hue="SAE",
        size=2,
        alpha=0.4,
        dodge=True, 
        jitter=True,
        legend=False,
        ax=ax_cos,
    )

    ax_cos.set_title("Cosine Similarity of true and reconstructed spike counts")
    ax_cos.set_ylabel("")
    ax_cos.set_xlabel("")
    ax_cos.set_ylim(0.1, 1.0)
    ax_cos.set_yticks(np.arange(0.1, 1.1, 0.1))
    
    # </ss>

    # </s>

    # <s> Log to wandb.

    # Log metrics figure
    wandb.log({"combined_metrics_plot": wandb.Image(fig)})
    plt.close(fig)
    
    # Log metrics values
    wandb.log({
        "r2_per_example_mean": np.mean(r2_per_example),
        "r2_per_unit_mean": np.mean(r2_per_unit),
        "cos_per_example_mean": np.mean(cos_sim_per_example),
        "cos_per_unit_mean": np.std(cos_sim_per_unit),
    })

    # </s>

def wandb_run(
    spk_cts: BFloat16[Tensor, "n_examples n_units"],
    sae_cfg: mt.SaeConfig,
    device: t.device
):
    """Start a wandb run in a sweep."""
    _run = wandb.init()
    
    # Training config
    n_epochs, batch_sz = wandb.config.epochs, wandb.config.batch_size
    n_steps = spk_cts.shape[0] // batch_sz * n_epochs
    log_freq = n_steps // n_epochs // 2
    neuron_resamples_per_epoch = 2
    neuron_resample_window = spk_cts.shape[0] // batch_sz // neuron_resamples_per_epoch

    # Set loss function according to current run config
    if "mse" in wandb.config.loss_fn:
        loss_fn = mt.mse
        lmse_tau = 0.0  # not used
    elif "lmse" in wandb.config.loss_fn:
        loss_fn = mt.lmse
        lmse_tau = float(wandb.config.loss_fn.split("_")[-1])
    
    # Train SAE with current run config
    sae = mt.Sae(sae_cfg).to(device)
    _data_log = optimize(
        spk_cts=spk_cts,
        model=sae,
        seq_len=sae_cfg.seq_len,
        loss_fn=loss_fn,
        lmse_tau=lmse_tau,
        lr=wandb.config.lr,
        use_lr_sched=wandb.config.use_lr_sched,
        neuron_resample_window=neuron_resample_window,
        batch_sz=batch_sz,
        n_steps=n_steps,
        log_freq=log_freq,
    )
    
    # Create post-run evaluation plots and metrics
    eval_model(spk_cts, sae)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SAE training sweep with WandB")
    parser.add_argument(
        "--spk-cts-file", required=True, type=str, help="Path to CSV file containing spike counts"
    )
    parser.add_argument(
        "--config-file", required=True, type=str, help="Path to YAML file containing sweep config"
    )
    args = parser.parse_args()
    
    # Get torch device info
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"{device=}")
    if device.type == "cuda":
        print(t.cuda.get_device_name(0))
        cur_mem_free = (
            t.cuda.get_device_properties(device).total_memory / 1e9
            - (t.cuda.memory_allocated(device) / 1e9) + (t.cuda.memory_reserved(device) / 1e9)
        )
        print(f"{cur_mem_free=:.2f} GB")

    # Load spike data
    counts_df = pd.read_csv(Path(args.spk_cts_file), index_col=0)
    n_input_ae = counts_df.shape[1]
    spk_cts = t.from_numpy(counts_df.to_numpy()).bfloat16().to(device)
    spk_cts /= spk_cts.max()  # max normalize spike counts

    # Load config
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract wandb config
    entity = config["wandb_config"]["entity"]
    project = config["wandb_config"]["project"]

    # Extract sweep config
    n_runs = config["sweep_config"]["n_runs"]

    # Create the sweep
    sweep_id = wandb.sweep(config, entity=entity, project=project)

    with wandb.init(project=project, job_type="config") as run:
        # Create static configuration artifact for dataset
        config_artifact = wandb.Artifact(
            "spike_counts_config",
            type="dataset",
            description="Model configuration for SAE sweep"
        )
        with config_artifact.new_file("config.json") as f:  # add config as file
            config_dict = {
                "spike_counts_file": args.spk_cts_file
            }
            f.write(json.dumps(config_dict, indent=4))
        run.log_artifact(config_artifact)
        # Create sweep progress tracker table
        progress_table = wandb.Table(columns=["runs: completed / total", "est seconds remaining"])
        progress_table.add_data(f"0 / {n_runs}", "Unknown")
        run.log({"sweep_progress": progress_table})
    
    # Define custom sweep agent function
    def sweep_agent():
        """Run the wandb sweep agent with custom run handling."""
        run = wandb.init()  # init run
        start_time = time.time()
        name = (
            f"d_sae={wandb.config.d_sae}  topk={wandb.config.top_k}  lr={wandb.config.lr:.1e}  "
            f"loss_fn={wandb.config.loss_fn}"
        )
        run.name = name
        run.save()  # save run name and then continue with run
        
        # Create the SAE cfg and run.
        sae_cfg = mt.SaeConfig(
            n_input_ae=n_input_ae,
            n_instances=wandb.config.n_instances,
            n_hidden_ae=wandb.config.d_sae,
            seq_len=wandb.config.seq_len,
            topk=wandb.config.top_k
        )
        wandb_run(spk_cts, sae_cfg, device)
        
        # Calculate run duration
        run_duration = time.time() - start_time
        wandb.log({"run_duration": run_duration})
        
        # Collect and format metrics
        metrics = {
            "name": run.name,
            "loss": f"{run.summary.get('loss', 0):.6f}",
            "l0_mean": f"{run.summary.get('l0_mean', 0):.1f}",
            "l0_std": f"{run.summary.get('l0_std', 0):.1f}",
            "r2_per_unit_mean": f"{run.summary.get('r2_per_unit_mean', 0):.3f}",
            "r2_per_example_mean": f"{run.summary.get('r2_per_example_mean', 0):.3f}",
            "cos_per_unit_mean": f"{run.summary.get('cos_per_unit_mean', 0):.3f}",
            "cos_per_example_mean": f"{run.summary.get('cos_per_example_mean', 0):.3f}",
        }

        # Log this run's data for the sweep summary table.
        sweep_summary_table = wandb.Table(
            columns=[
                "name",
                "final_loss",
                "r2_per_unit_mean",
                "r2_per_example_mean",
                "cos_sim_per_unit_mean",
                "cos_sim_per_example_mean",
                "overall_l0_mean",
                "overall_l0_std",
            ],
            data=[
                [
                    metrics["name"],
                    metrics["loss"],
                    metrics["r2_per_unit_mean"],
                    metrics["r2_per_example_mean"],
                    metrics["cos_per_unit_mean"],
                    metrics["cos_per_example_mean"],
                    metrics["l0_mean"],
                    metrics["l0_std"]
                ]
            ]
        )
        wandb.log({"sweep_summary_metrics": sweep_summary_table})
        
        # Update sweep progress tracker table
        sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
        completed_runs = len([r for r in sweep.runs if r.state == "finished"])
        avg_duration = sum(
            [r.summary.get("run_duration", 0) for r in sweep.runs if r.state == "finished"]
        ) / max(1, completed_runs)
        est_remaining = avg_duration * (n_runs - completed_runs)
        
        progress_table = wandb.Table(
            columns=["runs: completed / total", "est seconds remaining"],
            data=[[f"{completed_runs} / {n_runs}", f"{est_remaining:.1f}"]])
        wandb.log({"sweep_progress": progress_table})
        
        wandb.finish()  # finish run

    # Extract slurm config with defaults, and run sweep agent with slurm if configured
    slurm_config = config.get("slurm_config", {})
    if slurm_config:

        # Define slurm job function.
        def slurm_job(task_id: int, sweep_id: int, runs_per_task: int):
            """Runs a slurm job with the sweep agent."""
            os.environ["CUDA_VISIBLE_DEVICES"] = str(task_id % slurm_config["gpus_per_task"])
            wandb.login()
            wandb.agent(sweep_id, function=sweep_agent, count=runs_per_task)
        

        # Extract slurm parameters.
        log_dir = slurm_config.pop("log_dir")
        slurm_config["slurm_partition"] = slurm_config.get("slurm_partition", "gpu_branco")
        nodelist = slurm_config.pop("nodelist", None)
        slurm_config["tasks_per_node"] = slurm_config.get("tasks_per_node", 4)
        slurm_config["gpus_per_task"] = slurm_config.get("gpus_per_task", 1)
        slurm_config["cpus_per_task"] = slurm_config.get("cpus_per_task", 8)
        slurm_config["mem_gb"] = slurm_config.get("mem_gb", 128)
        # Set up executor.
        runs_per_task = math.ceil(n_runs / slurm_config["tasks_per_node"])  # upper limit
        log_dir = Path(log_dir) / f"sweep_{sweep_id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=log_dir)
        executor.update_parameters(**slurm_config)
        if nodelist:
            executor.update_parameters(slurm_additional_parameters={"nodelist": nodelist})
        # Submit tasks to executor.
        jobs = []
        for task_id in range(slurm_config["tasks_per_node"]):
            job = executor.submit(slurm_job, task_id, sweep_id, runs_per_task)
            jobs.append(job)
        print(f"Submitted {sweep_id=} split into {len(jobs)} jobs.")

        # Monitor job status.
        check_interval = 300  # seconds between checks
        while True:
            statuses = [job.done() for job in jobs]
            running_jobs = len(jobs) - sum(statuses)
            
            if running_jobs == 0:
                print("\nAll jobs ended.")
                break
                
            print(f"\n{running_jobs} jobs still running. Checking every {check_interval} seconds...")
            time.sleep(check_interval)
        
        # Collect and log results
        print("\nCollecting job results...")
        results = []
        for job in enumerate(jobs):
            try:
                result = job.result()
                results.append(result)
                print(f"{job}: {result}")
            except Exception as e:
                print(f"{job} failed with error: {e}")
                print(f"Traceback: {job.exception()}")

    else:
        # Start the sweep via the sweep agent locally.
        wandb.agent(sweep_id, function=sweep_agent, count=n_runs)

    # After all runs are complete, terminate the sweep.
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    sweep.state = "finished"
    sweep.update()
