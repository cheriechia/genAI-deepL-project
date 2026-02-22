import torch
import wandb

def save_best_model(model, model_name, mode, best_f1):
    # Store final best metric for sweep comparison
    wandb.summary["best_macro_f1"] = best_f1
    wandb.summary["model"] = model_name
    wandb.summary["mode"] = mode

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": dict(wandb.config),
        "model_name": model_name,
        "wandb_run_id": wandb.run.id
    }

    # Local save (baseline mode)
    if mode == "baseline":
        save_path = f"best_model_{model_name}.pt"
        torch.save(checkpoint, save_path)
        print(f"Saved best model locally to {save_path}")

    # Upload to W&B
    artifact = wandb.Artifact(
        name=f"{model_name}-{wandb.run.id}",
        type="model"
    )

    with artifact.new_file("model.pt", mode="wb") as f:
        torch.save(checkpoint, f)

    wandb.log_artifact(artifact)

    print(f"Saved best model to W&B artifact: {artifact.name}")
