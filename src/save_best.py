import torch
import wandb
import tempfile
import os

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
    artifact = wandb.Artifact(
        name=f"{model_name}-{wandb.run.id}",
        type="model"
    )

    # Save directly into artifact (Windows safe)
    with artifact.new_file("model.pt", mode="wb") as f:
        torch.save(model.state_dict(), f)

    wandb.log_artifact(artifact)

    print(f"Saved best model to W&B artifact: {artifact.name}")

    # Always upload to W&B
    # tmp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    # tmp_path = tmp_file.name
    # tmp_file.close()  # 🔥 IMPORTANT on Windows

    # torch.save(model.state_dict(), tmp_path)

    # artifact = wandb.Artifact(
    #     name=f"model-{wandb.run.id}",
    #     type="model"
    # )
    # artifact.add_file(tmp_path)
    # wandb.log_artifact(artifact)

    # print(f"Saved best model to W&B artifact {artifact.name}")
    # os.remove(tmp_path)  # clean up manually

    # # Save directly to W&B artifact
    # import tempfile
    # with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
    #     # Save model to temporary file
    #     torch.save(model.state_dict(), tmp_file.name)
    #     tmp_file.flush()  # ensure all data is written

    #     # Create artifact and add file **inside the block**
    #     artifact = wandb.Artifact(
    #         name=f"model-{wandb.run.id}",
    #         type="model"
    #     )
    #     artifact.add_file(tmp_file.name)  # use tmp_file.name here
    #     wandb.log_artifact(artifact)

    #     print(f"Saved best model to W&B artifact {artifact.name}")

        
    # with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
    #     torch.save(model.state_dict(), tmp_file.name)

    #     artifact = wandb.Artifact(
    #         name=f"model-{wandb.run.id}",
    #         type="model"
    #     )
    #     artifact.add_file(tmp_file.name)
    #     wandb.log_artifact(artifact)

    #     print(f"Saved best model to W&B artifact {artifact.name}")
