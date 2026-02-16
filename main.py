import argparse
import wandb
from src.bert import run_baseline as run_baseline_bert
from src.bert import run_sweep as run_sweep_bert
from src.cnn import run_baseline as run_baseline_cnn
from src.cnn import run_sweep as run_sweep_cnn
from src.mlp import run_baseline as run_baseline_mlp
from src.mlp import run_sweep as run_sweep_mlp
from src.lstm import run_baseline as run_baseline_lstm
from src.lstm import run_sweep as run_sweep_lstm
import multiprocessing

def launch_baseline(model_name, config_file, run_function, project):
    print(f"Running single training for model: {model_name}")
    run_function(project=project, config_file=config_file)

def launch_sweep(model_name, sweep_file, run_function, project):
    print(f"Launching sweep for {model_name}")
    sweep_id = wandb.sweep(sweep_file, project=project)
    wandb.agent(sweep_id, function=run_function)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "sweep"],
        help="Run a single training run or a W&B hyperparameter sweep"
    )
    parser.add_argument(
        "--model", type=str, default="bert",
        choices=["all", "bert", "mlp", "cnn", "lstm"],
        help="Select which model to train/run"
    )
    parser.add_argument(
        "--project", type=str, default="instagram-posts",
        help="W&B project name"
    )

    args = parser.parse_args()

    # Map model names to baseline run functions
    baselines = [
        ("bert", "bert_baseline.yaml", run_baseline_bert),
        ("cnn", "cnn_baseline.yaml", run_baseline_cnn),
        ("mlp", "mlp_baseline.yaml", run_baseline_mlp),
        ("lstm", "lstm_baseline.yaml", run_baseline_lstm),
    ]
    # Map model names to sweep files and run functions
    sweeps = [
        ("bert", "bert_sweep.yaml", run_sweep_bert),
        ("cnn", "cnn_sweep.yaml", run_sweep_cnn),
        ("mlp", "mlp_sweep.yaml", run_sweep_mlp),
        ("lstm", "lstm_sweep.yaml", run_sweep_lstm),
    ]

    # Hyperparameter sweep
    if args.mode == "sweep": 
        if args.model == "all":
            # Launch each sweep in a separate process
            processes = []
            for model_name, sweep_file, run_func in sweeps:
                p = multiprocessing.Process(
                    target=launch_sweep,
                    args=(model_name, sweep_file, run_func, args.project)
                )
                p.start()
                processes.append(p)

            # Wait for all sweeps to finish
            for p in processes:
                p.join()

        elif args.model == "bert":
            launch_sweep(args.model, "sweep_bert.yaml", run_sweep_bert, args.project)
            # sweep_id = wandb.sweep("sweep.yaml", project=args.project)
            # wandb.agent(sweep_id, function=run_sweep)
        elif args.model == "cnn":
            launch_sweep(args.model, "sweep_cnn.yaml", run_sweep_cnn, args.project)
            # sweep_id = wandb.sweep("cnn_sweep.yaml", project=args.project)
            # wandb.agent(sweep_id, function=run_sweep_cnn)
        elif args.model == "mlp":
            launch_sweep(args.model, "sweep_mlp.yaml", run_sweep_mlp, args.project)
        elif args.model == "lstm":
            launch_sweep(args.model, "sweep_lstm.yaml", run_sweep_lstm, args.project)
        
    # Single-run training
    else:
        print(f"Running single training for model: {args.model}")
        if args.model == "all":
            # Launch each sweep in a separate process
            processes = []
            for model_name, baseline_file, run_func in baselines:
                p = multiprocessing.Process(
                    target=launch_baseline,
                    args=(model_name, baseline_file, run_func, args.project)
                )
                p.start()
                processes.append(p)

            # Wait for all sweeps to finish
            for p in processes:
                p.join()
        elif args.model == "bert":
            launch_baseline(run_baseline_bert, "baseline_bert.yaml", project=args.project)
        elif args.model == "cnn":
            launch_baseline(run_baseline_cnn, "baseline_cnn.yaml", project=args.project)
        elif args.model == "mlp":
            launch_baseline(run_baseline_mlp, "baseline_mlp.yaml", project=args.project)
        elif args.model == "lstm":
            launch_baseline(run_baseline_lstm, "baseline_lstm.yaml", project=args.project)

if __name__ == "__main__":
    main()

# # Single-run BERT
# python src/main.py --mode single --model bert

# # Single-run CNN
# python src/main.py --mode single --model cnn

# # Start a sweep for BERT
# python src/main.py --mode sweep --model bert
