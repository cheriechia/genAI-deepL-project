import argparse
import wandb
import yaml
import os

from src.bert import run_baseline as run_baseline_bert
from src.bert import run_sweep as run_sweep_bert
from src.cnn import run_baseline as run_baseline_cnn
from src.cnn import run_sweep as run_sweep_cnn
from src.mlp import run_baseline as run_baseline_mlp
from src.mlp import run_sweep as run_sweep_mlp
from src.lstm import run_baseline as run_baseline_lstm
from src.lstm import run_sweep as run_sweep_lstm
from src.fusion import run_baseline as run_baseline_fusion
from src.fusion import run_sweep as run_sweep_fusion

def launch_baseline(model_name, config_file, run_function, project):
    print(f"Running baseline training for model: {model_name}")
    run_function(project=project, config_file=config_file)

def launch_sweep(model_name, sweep_file, run_function, project):
    print(f"Launching sweep for {model_name}")
    with open(sweep_file) as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, function=run_function, count=25) # limit sweeps to save time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="",
        choices=["baseline", "sweep"],
        help="Run a baseline training run or a W&B hyperparameter sweep"
    )
    parser.add_argument(
        "--model", type=str, default="",
        choices=["fusion", "all", "bert", "mlp", "cnn", "lstm"],
        help="Select which model to train/run"
    )
    parser.add_argument(
        "--project", type=str, default="instagram-posts",
        help="W&B project name"
    )

    args = parser.parse_args()

    # Map model names to baseline run functions
    baselines = [
        ("bert", "config/bert_baseline.yaml", run_baseline_bert),
        ("cnn", "config/cnn_baseline.yaml", run_baseline_cnn),
        ("mlp", "config/mlp_baseline.yaml", run_baseline_mlp),
        ("lstm", "config/lstm_baseline.yaml", run_baseline_lstm),
        ("fusion", "config/fusion_baseline.yaml", run_baseline_fusion)
    ]
    # Map model names to sweep files and run functions
    sweeps = [
        ("bert", "config/bert_sweep_frozen.yaml", run_sweep_bert),
        ("bert", "config/bert_sweep_unfrozen.yaml", run_sweep_bert),
        ("cnn", "config/cnn_sweep_frozen.yaml", run_sweep_cnn),
        ("cnn", "config/cnn_sweep_unfrozen.yaml", run_sweep_cnn),
        ("mlp", "config/mlp_sweep.yaml", run_sweep_mlp),
        ("lstm", "config/lstm_sweep.yaml", run_sweep_lstm),
        ("fusion", "config/fusion_sweep.yaml", run_sweep_fusion)
    ]

    if args.model == "fusion":
        if args.mode == "sweep":
            launch_sweep(model_name, sweep_file, run_func, args.project)
            # run_fusion(args.project)
        elif args.mode == "baseline":
            launch_sweep(model_name, sweep_file, run_func, args.project)
    else:
        # Hyperparameter sweep
        if args.mode == "sweep": 
            # Launch each sweep in a separate process
            for model_name, sweep_file, run_func in sweeps:
                if model_name == args.model: # if model name matches
                    launch_sweep(model_name, sweep_file, run_func, args.project)
            
        # Single-run training
        elif args.mode == "baseline":
            print(f"Running single training for model: {args.model}")
            # Launch each sweep in a separate process
            for model_name, baseline_file, run_func in baselines:
                if model_name == args.model: # if model name matches
                    launch_sweep(model_name, sweep_file, run_func, args.project)       

        else:
            print("No such mode")

    print("Finishing W&B run...")
    wandb.finish()
    print("Finished.")

    # Now all runs are done — safe to clean W&B cache
    os.system("wandb artifact cache cleanup")
    os.system("wandb gc --yes")

if __name__ == "__main__":
    main()

# # Single-run BERT
# python src/main.py --mode single --model bert

# # Single-run CNN
# python src/main.py --mode single --model cnn

# # Start a sweep for BERT
# python src/main.py --mode sweep --model bert
