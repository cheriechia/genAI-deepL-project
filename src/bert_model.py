# src/bert_model.py

import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel



def load_bert(model_name='bert-base-uncased'):
    """
    Load BERT model and BertTokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    return bert_model, tokenizer


class CaptionBERT(nn.Module):
    """
    BERT-based caption classifier with additional projection layer.

    Uses the [CLS] embedding for classification and optionally
    returns projected features for multimodal fusion.
    """

    def __init__(self, bert_model,
                 hidden_dim=256,
                 num_classes=3,
                 dropout=0.5):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(self.bert.config.hidden_size, hidden_dim) # self.bert.config.hidden_size is 768 for bert-base
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_emb)
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        if return_features:
            return x
        out = self.fc_out(x) # raw logits
        return out
