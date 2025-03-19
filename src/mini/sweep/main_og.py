"""Test wandb sweeps."""

# __main__
#   wandb.agent  # (local)
#     run_sweep_run
#   for job in jobs  # (remote)
#     executor.submit
#       run_wandb_agent
#         wandb.agent
#           run_sweep_run

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pandas as pd
import submitit
import torch as t
import yaml

from mini import train as mt
    

if __name__ == "__main__":
    # <s> Extract data and config and set up sweep.

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SAE training sweep with WandB")
    parser.add_argument(
        "--spk-cts-file", required=True, type=str, help="Path to CSV file containing spike counts"
    )
    parser.add_argument(
        "--config-file", required=True, type=str, help="Path to YAML file containing sweep config"
    )
    args = parser.parse_args()

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

    # </s>

    # <s> Define sweep run function.
    
    def run_sweep_run():
        """Called by wandb agent for each run in sweep."""
        device = t.device("cuda" if t.cuda.is_available() else "cpu")
        
        # Load spike counts from file.
        counts_df = pd.read_csv(Path(args.spk_cts_file), index_col=0)
        spk_cts = t.from_numpy(counts_df.to_numpy()).bfloat16().to(device)
        spk_cts /= spk_cts.max()  # max normalize spike counts
        print(f"Loaded spike counts: {spk_cts.shape}")
        
        # Initialize wandb run.
        run = wandb.init()
        start_time = time.time()
        run.name = (
            f"d_sae={wandb.config.d_sae}  topk={wandb.config.top_k}  lr={wandb.config.lr:.1e}  "
            f"loss_fn={wandb.config.loss_fn}"
        )
        run.save()  # save run name and then continue with run
        
        # Set training config
        n_epochs, batch_sz = wandb.config.epochs, wandb.config.batch_size
        n_steps = spk_cts.shape[0] // batch_sz * n_epochs
        log_freq = n_steps // n_epochs // 2
        neuron_resamples_per_epoch = 2
        neuron_resample_window = spk_cts.shape[0] // batch_sz // neuron_resamples_per_epoch
        
        # Create the SAE cfg and run training and evaluation.
        sae_cfg = mt.SaeConfig(
            n_input_ae=spk_cts.shape[1],
            n_instances=wandb.config.n_instances,
            n_hidden_ae=wandb.config.d_sae,
            seq_len=wandb.config.seq_len,
            topk=wandb.config.top_k
        )
        sae = mt.Sae(sae_cfg).to(device)
        _data_log = mt.optimize(  # train model
            spk_cts=spk_cts,
            model=sae,
            seq_len=sae_cfg.seq_len,
            loss_fn=wandb.config.loss_fn,
            lr=wandb.config.lr,
            use_lr_sched=wandb.config.use_lr_sched,
            neuron_resample_window=neuron_resample_window,
            batch_sz=batch_sz,
            n_steps=n_steps,
            log_freq=log_freq,
        )
        mt.eval_model(spk_cts, sae)  # evaluate model
        
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

        # Log this run's data for the sweep progress table.
        sweep = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}")
        completed_runs = len([r for r in sweep.runs if r.state == "finished"])
        avg_duration = sum(
            [r.summary.get("run_duration", 0) for r in sweep.runs if r.state == "finished"]
        ) / max(1, completed_runs)
        est_remaining = avg_duration * (n_runs - completed_runs)
        progress_table = wandb.Table(
            columns=["runs: completed / total", "est seconds remaining"],
            data=[[f"{completed_runs} / {n_runs}", f"{est_remaining:.1f}"]]
        )
        wandb.log({"sweep_progress": progress_table})
        
        # wandb.finish()  # finish run

    # </s>

    # <s> Run sweep.

    # <ss> Run sweep as slurm multi-job.

    slurm_config = config.get("slurm_config", {})
    if slurm_config:
        # Extract slurm parameters.
        log_dir = slurm_config.pop("log_dir")
        partition = slurm_config.pop("slurm_partition", "gpu_branco") 
        nodelist = slurm_config.pop("nodelist", None)
        n_tasks = slurm_config.pop("n_tasks", 4)
        gpus_per_task = slurm_config.pop("gpus_per_task", 1)
        cpus_per_task = slurm_config.pop("cpus_per_task", 8)
        mem_gb = slurm_config.pop("mem_gb", 256)
        job_name = slurm_config.pop("slurm_job_name", f"sweep_{sweep_id}")

        # Define function to be called by submitit executor.
        def run_wandb_agent_slurm(task_id):
            """Called by submitit executor to run wandb agent, once per task."""
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set specific GPU if needed
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_START_METHOD"] = "thread"
            os.environ["PYTHONUNBUFFERED"] = "1"
            os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
            os.environ["WANDB_SERVICE_PORT"] = str(8765 + task_id)
            
            # This single agent will pull configs from the sweep controller
            wandb.agent(
                sweep_id,
                function=run_sweep_run,
                entity=entity,
                project=project,
                count=(n_runs // n_tasks) + 1,  # max runs per agent
            )

        # Create and call `n_tasks` sweep agents (jobs) that belong to same sweep & run in parallel
        log_dir_path = Path(log_dir) / f"sweep_{sweep_id}"
        log_dir_path.mkdir(parents=True, exist_ok=True)
        jobs = []
        for i in range(n_tasks):
            # Set up job.
            executor = submitit.AutoExecutor(folder=log_dir_path)
            executor.update_parameters(
                name=job_name,
                slurm_partition=partition,
                nodes=1,
                tasks_per_node=1,
                slurm_cpus_per_task=cpus_per_task,
                slurm_gpus_per_task=gpus_per_task,
                mem_gb=mem_gb,
            )
            if nodelist:
                executor.update_parameters(slurm_additional_parameters={"nodelist": nodelist})
            # Submit job.
            job = executor.submit(run_wandb_agent_slurm, i)
            print(
                f"wandb sweep ({sweep_id=}) running via slurm job ({job.job_id}); "
                f"task {i+1}/{n_tasks}"
            )

    # </ss>

    # <ss> Run sweep locally.

    else:


        def run_wandb_agent_local(rank):
            # Set up reqs for GPU for this process
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            os.environ["WANDB_START_METHOD"] = "thread"
            os.environ["WANDB_SERVICE_PORT"] = str(8765 + rank)
            
            # This agent will pull configs from the sweep controller
            wandb.agent(
                sweep_id, 
                function=run_sweep_run,
                entity=entity,
                project=project,
                count=(n_runs // n_gpus) + 1,  # distribute runs across GPUs
            )

        
        n_gpus = t.cuda.device_count()
        if n_gpus > 1:  # run one wandb agent process per gpu
            print(f"Found {n_gpus} GPUs available for parallel sweep execution")

            processes = []
            for gpu_id in range(n_gpus):
                print(f"Starting wandb agent on GPU {gpu_id}")
                p = mp.Process(target=run_wandb_agent_local, args=(gpu_id,))
                p.start()
                processes.append(p)
            
            for p in processes:  # wait for all processes to initialize before joining
                p.join()
        
        else:
            run_wandb_agent_local(0)  # single gpu or cpu
    
    # </ss>

    # </s>
