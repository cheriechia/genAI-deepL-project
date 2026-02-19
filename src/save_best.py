import torch
import wandb
import tempfile


def save_best_model(model, model_name, mode, best_f1):
    # Store final best metric for sweep comparison
    wandb.summary["best_macro_f1"] = best_f1
    wandb.summary["model"] = model_name
    wandb.summary["mode"] = mode

    if mode == "baseline": # local save
        save_path = f"best_model_{model_name}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to local {save_path}")

    # upload to wandb:
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        torch.save(model.state_dict(), tmp_file.name)

        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model"
        )
        artifact.add_file(tmp_file.name)
        wandb.log_artifact(artifact)

        print(f"Saved best model to W&B artifact {artifact.name}")
