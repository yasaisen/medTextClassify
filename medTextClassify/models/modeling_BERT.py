"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506171442
"""

import torch.nn as nn
from transformers import AutoModel


class BioBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.85):
        super(BioBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        hidden = self.relu(self.hidden(pooled_output))
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)
        return logits

