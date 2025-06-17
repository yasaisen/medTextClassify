"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506171442
"""

import torch
from torch.utils.data import Dataset, DataLoader
import re

from ..common.utils import load_json_data


class MedicalTextDataset(Dataset):
    def __init__(self, 
            tokenizer, 
            data_path='./data_2506091959.json', 
            max_length=512, 
            split='train',
        ):
        self.art_dataset = []
        _art_dataset = load_json_data(data_path)
        
        if not _art_dataset:
            raise ValueError(f"Unable to load data or data is empty:{data_path}")
        
        for sample in _art_dataset:
            if sample.get('split') == split:
                required_fields = ['pmid', 'title', 'abstract', 'label']
                if all(field in sample for field in required_fields):
                    self.art_dataset.append(sample)
                else:
                    print(f"Warning: The sample is missing required fields, skipping this sample")

        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Load {split} data: {len(self.art_dataset)} samples")
    
    def __len__(self):
        return len(self.art_dataset)
    
    def pre_caption(self, caption: str):
        if not isinstance(caption, str):
            caption = str(caption)
        
        caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
        caption = re.sub(r"\s{2,}", " ", caption)
        caption = caption.rstrip("\n").strip(" ")

        if self.max_length and len(caption) > self.max_length * 4:
            caption = caption[:self.max_length * 4]

        return caption

    def __getitem__(self, idx):
        try:
            sample = self.art_dataset[idx]
            pmid = sample['pmid']
            title = sample.get('title', '')
            abstract = sample.get('abstract', '')
            label = sample['label']

            if not isinstance(label, int) or label < 0:
                print(f"Warning: Invalid label value {label}, set to 0")
                label = 0

            text = self.pre_caption(f"{title} {abstract}")

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'pmid': pmid,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error: Error processing sample {idx} - {e}")
            return {
                'pmid': 'error',
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.tensor(0, dtype=torch.long)
            }

