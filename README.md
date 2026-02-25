# genAI-deepL-project

## Table of Contents
* [Problem Statement](#problem-statement)
* [Overview of the folder structure and description of files](#overview-of-the-folder-structure-and-description-of-files)
* [Description of logical steps/flow](#description-of-logical-stepsflow)
* [Dataset summary](#dataset-summary)
* [How the features in the dataset are processed](#describe-how-the-features-in-the-dataset-are-processed)
* [Explanation of choice of models](#explanation-of-choice-of-models)
* [Hyperparameters](#hyperparameters)
* [Evaluation of the models developed](#evaluation-of-the-models-developed)
    * [LSTM VS Frozen BERT VS Unfrozen BERT](#lstm-vs-frozen-bert-vs-unfrozen-bert)
    * [Frozen ResNet18 vs Unfrozen ResNet18](#frozen-resnet18-vs-unfrozen-resnet18)
    * [All the best models, and fusion](#all-the-best-models-and-fusion)
* [Comparing Fusion Models](#comparing-fusion-models)
* [Ablation Study](#ablation-study)
* [Conclusions](#conclusions)
* [Possible improvement](#possible-improvement)

## Problem statement
This project addresses the business problem of predicting the engagement performance of Instagram posts using historical post data. 

Engagement is defined as the level of audience interaction a post receives, measured through likes and comments relative to follower count. By predicting engagement levels in advance, businesses can assess their own content or influencer posts, enabling more informed decisions in content planning and marketing campaigns while reducing risk, improving return on investment, and reducing time spent on manual and subjective inspection of historical posts, captions, images and engagement metrics.

## Overview of the folder structure and description of files
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
├── README.md
├── requirements.txt
└── main.py                                 [Main launcher of all runs]
```

## Description of logical steps/flow of the pipeline
![flowchart](charts/genAIdeepL.drawio.png)

## Dataset summary
2 datasets were considered for this project, and they both provide the following features originally:

Feature|Data type
---|---
user_id    |        object 
followers    |     int64  
following      |     int64  
publish_timestamp |  object 
has_location    |     int64  
is_carousel    |    int64  
num_images     |      int64  
is_sponsored    |    int64  
image_path     |      object 
caption       |      object 
likes         |      int64  
comments      |      int64  
engagement_rate  |    float64

The smaller dataset has 1968 entries, while I extracted 100k random entries from the larger dataset.

## How the features in the dataset are processed
The table below shows the final dataset after some data engineering:

Attribute|Extracted/Engineered | Description | Data type | Usage
---|---|---|---|---
user_id|Extracted|ID of the post’s owner|int64|For train test split only
following|Extracted|Number of accounts owner is following|int64|MLP
publish_timestamp|Extracted|UTC time of publishing post|datetime64[ns]|For train test split only
has_location|Extracted|Whether the post has location or not|int64|MLP
is_carousel|Extracted|Whether it post has multiple images or not|int64|MLP
num_images|Extracted|Number of images in the post\int64\MLP
is_sponsored|Extracted|Whether the post is labelled as Paid Partnership or not|int64|MLP
image_path|Extracted|Path to image (or 1st image in carousel)|object|CNN
caption|Extracted|Post caption|object|RNN
follower_following_ratio|Engineered|Number of followers divided by number of accounts following|float64|MLP
hour|Engineered|Hour of publishing post (UTC)|int64|MLP
day|Engineered|Day of publishing post|object|MLP
is_weekend|Engineered|Whether the post was published on a weekend or not|bool|MLP
caption_word_count|Engineered|Number of words in caption, excluding hashtags|int64|MLP
hashtags|Extracted|List of hashtags|object|Unused
num_hashtags|Engineered|Number of hashtags|int64|MLP

Output: engagement_label - a low, moderate or high engagement label on the Instagram post. Split into 3 balanced classes based on count in each class.
Label|Small dataset | Large dataset
---|---|---
Low | less than 0.006% | less than 0.02%
Moderate | 0.006% to 0.02% | 0.02% to 0.04%
High | Above 0.02% | Above 0.04%

## Explanation of choice of models
A multimodal deep learning architecture is proposed, combining separate networks for different data modalities: 
* Multilayer Perceptron for metadata: This captures non-linear interactions and ensures compatibility for later fusion with text and image modules.
* LSTM or BERT for captions. 
    * LSTM: Word order and context are preserved through recurrent sequence processing and hidden state propagation across time steps.  
    * BERT: Evaluated to leverage pretrained bidirectional context and richer semantic representations.
* ResNet18 for images: Shallower depth reduces overfitting compared to deeper variants like ResNet50

## Hyperparameters
Hyperparameters were tuned with the help of WandB Sweeps using the Bayes method.
Parameters | MLP | LSTM | BERT (frozen) | BERT (unfrozen) | ResNet18 (frozen) | ResNet18 (unfrozen) 
---|---|---|---|---|---|---
hidden_dim | Hd1: [32, 64, 128], Hd2: [16, 32, 64] | [32, 64, 128] | 256 | 256 | - | - 
embed_dim | - | [50, 100] | - | - | - | - 
dropout | min: 0.3, max: 0.5 | min: 0.3, max: 0.6 | min: 0.3, max: 0.5 | min: 0.3, max: 0.5 | min: 0.3, max: 0.5 | min: 0.3, max: 0.5 |
max_len | - | [32, 64, 128] | [64, 128] | [64, 128] | - | -
learning_rate | min: 0.0005, max: 0.001 | min: 0.0005, max: 0.001 | min: 0.0005, max: 0.001 | min: 0.00002, max: 0.00005 | lr_head min: 0.0005, max: 0.002 | lr_backbone: min: 0.0001, max: 0.001, lr_head min: 0.0005, max: 0.002 
freeze | - | - | true | false | true | false

## Evaluation of the sub-models developed
### LSTM VS Frozen BERT VS Unfrozen BERT
<img src="charts/testF1_W&B Chart 23_02_2026, 18_55_04.png" width="48%" /><img src="charts/trainF1_W&B Chart 23_02_2026, 18_56_02.png" width="48%" />
* After sweeps, LSTM overfit more and performed worse than frozen BERT, leading to the selection of BERT for captions.
* Frozen and unfrozen BERT had similar test macro F1
* Unfrozen BERT overfit (train 69.7%, test 63%), while frozen BERT achieved comparable test performance (train 60%, test 63%) suggesting stable generalization.
* Frozen BERT is the final model for captions, with best macro-F1 score of 63%

### Frozen ResNet18 vs Unfrozen ResNet18
<img src="charts/testF1_W&B Chart 23_02_2026, 19_23_17.png" width="48%" /><img src="charts/trainF1_W&B Chart 23_02_2026, 19_23_48.png" width="48%" />
* In both test and train, the unfrozen ResNet18 performed better
* There was little to no overfitting on ResNet18, whether frozen or not
* Unfrozen ResNet18 had little to no overfitting (train 62%, test 59.8%), while frozen ResNet18 achieved comparable test performance (train 56%, test 58%), suggesting stable generalization.
* Unfrozen ResNet18 is the final model for images, with best macro-F1 score of 59.8%

### All the best models, and fusion
* The best MLP has macro-F1 score of 59%
* The overall Fusion Model has best macro-F1 score of 68.6%
<img src="charts/testF1_W&B Chart 23_02_2026, 19_26_10.png" width="48%" /><img src="charts/trainF1_W&B Chart 23_02_2026, 19_26_31.png" width="48%" />
* Mild overfitting in fusion model is observed from the drop in macro-F1 from train to test (72% to 68%)
* The fusion model did help to improve the macro-F1 score, but the final performance is still only moderate.

## Comparing Fusion Models
<img src="charts/allFusion_testF1_W&B Chart 25_02_2026, 15_19_55.png" width="48%" /><img src="charts/allFusion_trainF1_W&B Chart 25_02_2026, 15_19_55.png" width="48%" />
* The basic 1-layer fusion model performed the best at inference time
* It seemed like the 2-layer fusion model performed slightly worse, and the basic 1-layer fusion model overfit the most when rerun, but this could be attributed to randomness.
* Additional layers in fusion model did not help, likely because of a dataset size limitation, or that models already produce high level embeddings, or because features went through a bottleneck layer before fusion.

## Ablation study
* This is to satisfy my curiosity on which models - BERT, ResNet18 or MLP - had the most positive or negative influence on the fused model outcome.

<img src="charts/ablation_testF1_W&B Chart 25_02_2026, 19_17_01.png" width="48%" /><img src="charts/ablation_trainF1_W&B Chart 25_02_2026, 19_17_01.png" width="48%" />

* Interestingly, the fusion model performed the best when it fused all models, or only excluded ResNet18. This means that ResNet18 had almost no influence on predicting engagement labels, and BERT and MLP had similar contribution.

## Conclusions
* At 59% to 63% for macro-F1, each model was better than the random-guess performance of 33% (for 3 classes), showing that each model performs boderline moderately, with BERT and captions being slightly more useful for predicting engagement labels
* At 68% macro-F1 for the fusion model, this shows that the combination of features does help with engagement label prediction. The fusion model performs decently but is not the best.

<img src="charts/test_CM.png" width="80%" />

* The confusion matrix shows that many samples with Medium/Moderate engagement were predicted as Low, and vice versa. 
* The Medium/Moderate engagement label also has the fewest correct predictions.

## Possible improvement
* Larger dataset with wider spread of engagement rates
* Labelling if engagement rate was higher than a certain threshold, instead of 3 classes
* Add in use of hashtags that are currently unused
* More intermediate FC layers before classifier head for more gradual bottleneck
* Different way to merge the models instead of just FC layers: tried 2 FC layers, tried gating.
* Fuse features earlier, before the bottleneck layer (pretrained model → dropout → fuse)
* Using multimodal transformers instead, for cross-modal relationships to be learned through multiple layers
* Widen hyperparameter sweep boundaries
