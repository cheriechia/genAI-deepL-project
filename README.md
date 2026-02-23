# genAI-deepL-project

## Table of Contents

* [a. Overview of the submitted folder and the folder structure](#a-overview-of-the-submitted-folder-and-the-folder-structure)
* [b. Description of the files used in the project](#b-description-of-files-used-in-the-project)
* [d. Description of logical steps/flow of the pipeline](#d-description-of-logical-stepsflow-of-the-pipeline)
* [e. Overview of key findings from the EDA and the choices made in the pipeline](#e-overview-of-key-findings-from-the-eda-and-the-choices-made-in-the-pipeline)
* [f. Describe how the features in the dataset are processed](#f-describe-how-the-features-in-the-dataset-are-processed)
* [g. Explanation of your choice of models](#g-explanation-of-your-choice-of-models)
* [h. Evaluation of the models developed](#h-evaluation-of-the-models-developed)

## a. Overview of the submitted folder and the folder structure.
```
├── src/
│   ├── bert.py                         [Prepare data, run baseline/sweep of BERT]
│   ├── bert_dataset.py                 [TextDataset for BERT, and dataloader]
│   ├── bert_model.py                   [CaptionBERT model]
│   ├── cnn.py                          [Prepare data, run baseline/sweep of ResNet18]
│   ├── cnn_dataset.py                  [ImageDataset for CNN, and dataloader]
│   ├── cnn_model.py                    [ImageResNet ResNet18 model]
│   ├── config.py                       [General config (device, seed, num_classes etc.)]
│   ├── evaluate_metrics.py             [General evaluation of accuracy, macroF1, confusion matrix]
│   ├── fusion.py                       [Functions to get best run from WandB, load best models, run baseline/sweep of fusion model]
│   ├── fusion_model.py                 [FusionModel model]
│   ├── lstm.py                         [Prepare data, run baseline/sweep of LSTM]
│   ├── lstm_dataset.py                 [CaptionDataset for LSTM, and dataloader]
│   ├── lstm_model.py                   [CaptionRNN model]
│   ├── mlp.py                          [Prepare data, run baseline/sweep of MLP]
│   ├── mlp_dataset.py                  [MetadataDataset for MLP, and dataloader]
│   ├── mlp_model.py                    [MetadataMLP model]
│   ├── precompute_fusion_features.py   [Load best models runs, prepare data, save as .pt for fusion run.]
│   ├── save_best.py                    [General saving of best model from each baseline/sweep run, upload to WandB]
│   ├── train.py                        [General train, eval, wandb metrics logging manager]
│   └── utils.py                        [General set seed for repeatability and compute weights for class balance]
|
├── config/                             [config for baseline/sweep runs as named]
│   ├── bert_baseline.yaml
│   ├── bert_sweep_frozen.yaml
│   ├── bert_sweep_unfrozen.yaml
│   ├── cnn_baseline.yaml
│   ├── cnn_sweep_frozen.yaml
│   ├── cnn_sweep_unfrozen.yaml
│   ├── fusion_baseline.yaml
│   ├── fusion_selected_runs.yaml       [final selected best run IDs from WandB]
│   ├── fusion_sweep.yaml
│   ├── lstm_baseline.yaml
│   ├── lstm_sweep.yaml
│   ├── mlp_baseline.yaml
│   └── mlp_sweep.yaml
|
├── noteboooks/
│   ├── cnn_for_images.ipynb                [Initial CNN tests (milestone)]
│   ├── dataset_filtering_and_pkl.ipynb     [Preparation of large dataset to match formatting of small one]
│   ├── eda_preprocessing.ipynb             [Preprocessing, EDA of small dataset]
│   ├── eda_preprocessing_large.ipynb       [Preprocessing, EDA of large dataset]
│   ├── mlp_for_metedata.ipynb              [Initial MLP tests (milestone)]
│   ├── prepare_test_data_huggingface.ipynb [Random filter for small dataset on HF]
│   ├── rnn_for_captions_BERT.ipynb         [Initial BERT tests (milestone)]
│   └── rnn_for_captions_LSTM.ipynb         [Initial LSTM tests (milestone)]
|
├── wandb export/                           [All exported data and charts]
├── README.md
├── requirements.txt
└── main.py                                 [Main launcher of all runs]
```

## b. Description of files used in the project