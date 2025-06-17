"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506171442
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: Data file not found {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed - {e}")
        return []

class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.learning_rates = []
        
    def update_train(self, loss, accuracy, lr):
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(lr)
        
    def update_val(self, loss, accuracy, f1_score):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        self.val_f1_scores.append(f1_score)
        
    def save_metrics(self, filepath):
        metrics = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'learning_rates': self.learning_rates
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

def plot_improved_metrics(metrics_tracker, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(metrics_tracker.train_losses, label='Train Loss', color='blue', linewidth=2)
    axes[0, 0].plot(metrics_tracker.val_losses, label='Val Loss', color='red', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(metrics_tracker.train_accuracies, label='Train Acc', color='blue', linewidth=2)
    axes[0, 1].plot(metrics_tracker.val_accuracies, label='Val Acc', color='red', linewidth=2)
    axes[0, 1].set_title('Accuracy Curves', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(metrics_tracker.val_f1_scores, label='Val F1 Score', color='green', linewidth=2)
    axes[0, 2].set_title('F1 Score', fontsize=14)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    train_val_gap = np.array(metrics_tracker.train_accuracies) - np.array(metrics_tracker.val_accuracies)
    axes[1, 0].plot(train_val_gap, label='Accuracy Gap', color='orange', linewidth=2)
    axes[1, 0].set_title('Overfitting Analysis', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train - Val Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    
    axes[1, 1].plot(metrics_tracker.learning_rates, label='Learning Rate', color='purple', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(metrics_tracker.val_accuracies, label='Val Acc', color='red', linewidth=2)
    axes[1, 2].plot(metrics_tracker.val_f1_scores, label='Val F1', color='green', linewidth=2)
    axes[1, 2].set_title('Validation Metrics', fontsize=14)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics.png'), dpi=300, bbox_inches='tight')
    plt.show()


















