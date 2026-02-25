# precompute_fusion_features.py

import torch
import yaml
import pandas as pd
from tqdm import tqdm

from src.config import DEVICE, SEED
from src.utils import set_seed
from src.fusion import get_run_by_id, load_best_bert, load_best_cnn, load_best_mlp

from src.bert import data_preparation as data_prep_bert
from src.mlp import data_preparation as data_prep_mlp
from src.cnn import data_preparation as data_prep_cnn

from src.bert_dataset import create_dataloaders as dataloader_bert
from src.mlp_dataset import create_dataloaders as dataloader_mlp
from src.cnn_dataset import create_dataloaders as dataloader_cnn

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def extract_features(model_name):
    set_seed(SEED)

    # -------------------------
    # Load dataset
    # -------------------------
    train_df = pd.read_csv("data/train_df.csv", parse_dates=["publish_timestamp"])
    test_df  = pd.read_csv("data/test_df.csv", parse_dates=["publish_timestamp"])

    y_train = torch.tensor(train_df["engagement_label"].values, dtype=torch.long)
    y_test  = torch.tensor(test_df["engagement_label"].values, dtype=torch.long)

    # -------------------------
    # Data preparation
    # -------------------------

    # BERT
    with open("config/fusion_selected_runs.yaml", "r") as f:
        run_ids = yaml.safe_load(f)

    best_bert_run = get_run_by_id(run_id=run_ids["bert"])
    bert_model, tokenizer, bert_config = load_best_bert(best_bert_run)
    bert_model.eval()

    train_encodings, test_encodings = data_prep_bert(
        train_df,
        test_df,
        tokenizer,
        bert_config["max_len"]
    )

    # MLP
    X_train_proc, X_test_proc, y_train, y_test = data_prep_mlp(train_df, test_df)
    

    # CNN
    train_transform, test_transform = data_prep_cnn(train_df, test_df)
    

    # -------------------------
    # Save inputs to indiv models
    # -------------------------
    if model_name != 'fusion': 
        # BERT inputs
        # torch.save(train_encodings, "features/bert/bert_train_inputs.pt")
        # torch.save(test_encodings,  "features/bert/bert_test_inputs.pt")
        torch.save(
            {
                "X_train": train_encodings,
                "y_train": y_train
            },
            "features/bert/bert_train_inputs.pt"
        )
        torch.save(
            {
                "X_test": test_encodings,
                "y_test": y_test
            },
            "features/bert/bert_test_inputs.pt"
        )

        # MLP inputs
        # torch.save(X_train_proc, "features/mlp/mlp_train_inputs.pt")
        # torch.save(X_test_proc,  "features/mlp/mlp_test_inputs.pt")
        torch.save(
            {
                "X_train": X_train_proc,
                "y_train": y_train
            },
            "features/mlp/mlp_train_inputs.pt"
        )
        torch.save(
            {
                "X_test": X_test_proc,
                "y_test": y_test
            },
            "features/mlp/mlp_test_inputs.pt"
        )



        # CNN inputs (if dataset-style transforms)
        # torch.save(train_transform, "features/cnn/cnn_train_meta.pt")
        # torch.save(test_transform,  "features/cnn/cnn_test_meta.pt")
        torch.save(
            {
                "X_train": train_transform,
                "y_train": y_train
            },
            "features/cnn/cnn_train_meta.pt"
        )
        torch.save(
            {
                "X_test": test_transform,
                "y_test": y_test
            },
            "features/cnn/cnn_test_meta.pt"
        )

    elif model_name == 'fusion':
        # -------------------------
        # Load selected run IDs
        # -------------------------
        # MLP
        best_mlp_run  = get_run_by_id(run_id=run_ids["mlp"])
        mlp_model, mlp_config = load_best_mlp(best_mlp_run, input_dim=X_train_proc.shape[1]) # Input size matching one-hot expanded features
        mlp_model.eval()

        # CNN
        best_cnn_run  = get_run_by_id(run_id=run_ids["cnn"])
        cnn_model, cnn_config = load_best_cnn(best_cnn_run)
        cnn_model.eval()
        # with open("config/fusion_selected_runs.yaml", "r") as f:
        #     run_ids = yaml.safe_load(f)

        # # best_bert_run = get_run_by_id(run_id=run_ids["bert"])
        # best_cnn_run  = get_run_by_id(run_id=run_ids["cnn"])
        # best_mlp_run  = get_run_by_id(run_id=run_ids["mlp"])

        # # bert_model, tokenizer, bert_config = load_best_bert(best_bert_run)
        # cnn_model, cnn_config = load_best_cnn(best_cnn_run)
        # mlp_model, mlp_config = load_best_mlp(best_mlp_run, input_dim=X_train_proc.shape[1]) # Input size matching one-hot expanded features

        # # bert_model.eval()
        # cnn_model.eval()
        # mlp_model.eval()

        # reassign labels
        # y_train = torch.tensor(y_train, dtype=torch.long)
        # y_test  = torch.tensor(y_test, dtype=torch.long)

        # set seed for dataloader shuffling order to make it deterministic
        g = torch.Generator().manual_seed(SEED)

        bert_train_loader, bert_test_loader = dataloader_bert(
            train_encodings, test_encodings,
            y_train, y_test,
            bert_config["batch_size"], g,
            shuffle_train=False   # don't mix training labels
        )

        mlp_train_loader, mlp_test_loader = dataloader_mlp(
            X_train_proc, X_test_proc,
            y_train, y_test,
            mlp_config["batch_size"], g,
            shuffle_train=False   # don't mix training labels
        )

        cnn_train_loader, cnn_test_loader = dataloader_cnn(
            train_df, test_df,
            train_transform, test_transform,
            cnn_config["batch_size"], g,
            shuffle_train=False   # don't mix training labels
        )

        # --------------------------------------------------
        # FEATURE EXTRACTION
        # --------------------------------------------------

        def compute_split(bert_loader, cnn_loader, mlp_loader):
            all_features = []

            with torch.no_grad():
                for bert_batch, cnn_batch, mlp_batch in tqdm(
                    zip(bert_loader, cnn_loader, mlp_loader),
                    total=len(bert_loader)
                ):
                    # BERT
                    input_ids = bert_batch["input_ids"].to(DEVICE)
                    attention_mask = bert_batch["attention_mask"].to(DEVICE)

                    bert_feat = bert_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_features=True
                    )

                    # CNN
                    images, _ = cnn_batch
                    images = images.to(DEVICE)
                    cnn_feat = cnn_model(images, return_features=True)

                    # MLP
                    metadata, _ = mlp_batch
                    metadata = metadata.to(DEVICE)
                    mlp_feat = mlp_model(metadata, return_features=True)

                    fusion_feat = torch.cat(
                        [bert_feat, cnn_feat, mlp_feat],
                        dim=1
                    )

                    # for ablation study
                    # fusion_feat = torch.cat(
                    #     [cnn_feat, mlp_feat],
                    #     dim=1
                    # )

                    all_features.append(fusion_feat.cpu())

            return torch.cat(all_features, dim=0)

        print("Extracting TRAIN features...")
        X_train_fusion = compute_split(
            bert_train_loader,
            cnn_train_loader,
            mlp_train_loader
        )

        print("Extracting TEST features...")
        X_test_fusion = compute_split(
            bert_test_loader,
            cnn_test_loader,
            mlp_test_loader
        )

        # -------------------------
        # Save to disk
        # -------------------------
        torch.save(X_train_fusion, "features/fusion/X_train_fusion.pt")
        torch.save(X_test_fusion, "features/fusion/X_test_fusion.pt")
        torch.save(y_train, "features/fusion/y_train_fusion.pt")
        torch.save(y_test, "features/fusion/y_test_fusion.pt")

        print("✅ Fusion features saved successfully.")

