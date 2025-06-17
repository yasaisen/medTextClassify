"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506171442
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch import optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..models.modeling_BERT import BioBERTClassifier
from ..datasets.medTextSet import MedicalTextDataset
from ..common.utils import MetricsTracker, plot_improved_metrics


def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    # model.train()
    model.classifier = model.classifier.train()
    model.hidden = model.hidden.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions/total_predictions:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

def validate(model, dataloader, device, criterion):
    # model.eval()
    model.classifier = model.classifier.eval()
    model.hidden = model.hidden.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
    except:
        f1 = 0.0
    
    return avg_loss, accuracy, f1, all_predictions, all_labels

def train_biobert_classifier():
    MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    NUM_CLASSES = 2
    MAX_LENGTH = 512
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    
    nowtime = datetime.now().strftime("%y%m%d%H%M")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print("Loading BioBERT tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = BioBERTClassifier(MODEL_NAME, NUM_CLASSES).to(device)

        for name, param in model.bert.named_parameters():
            param.requires_grad = False
        model.bert = model.bert.eval()
        model.classifier = model.classifier.train()
        model = model.train()
        
        print("Creating dataset...")
        train_dataset = MedicalTextDataset(
            tokenizer=tokenizer, 
            data_path='./dataset/data_2506101500.json', 
            max_length=MAX_LENGTH, 
            split='train',
        )
        val_dataset = MedicalTextDataset(
            tokenizer=tokenizer, 
            data_path='./dataset/data_2506101500.json', 
            max_length=MAX_LENGTH, 
            split='valid',
        )
        
        if len(train_dataset) == 0:
            raise ValueError("train_dataset is null")
        if len(val_dataset) == 0:
            raise ValueError("val_dataset is null")
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )
        
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.9 * total_steps),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        metrics_tracker = MetricsTracker()
        
        best_val_f1 = 0

        output_dir = os.path.join(os.getcwd(), 'outputs', nowtime)
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, f"best_biobert_classifier_{nowtime}.pth")
        
        print("Starting training...")
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("-" * 50)
            
            train_loss, train_acc = train_epoch(
                model, train_dataloader, optimizer, scheduler, device, criterion
            )
            
            val_loss, val_acc, val_f1, val_preds, val_true = validate(
                model, val_dataloader, device, criterion
            )
            
            current_lr = scheduler.get_last_lr()[0]
            metrics_tracker.update_train(train_loss, train_acc, current_lr)
            metrics_tracker.update_val(val_loss, val_acc, val_f1)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_f1': best_val_f1,
                    'model_config': {
                        'model_name': MODEL_NAME,
                        'num_classes': NUM_CLASSES,
                        'max_length': MAX_LENGTH
                    }
                }, best_model_path)
                print(f"New best model saved with F1 score: {best_val_f1:.4f}")
        
        metrics_tracker.save_metrics(os.path.join(output_dir, 'training_metrics.json'))
        plot_improved_metrics(metrics_tracker, output_dir)

        final_val_loss, final_val_acc, final_val_f1, final_preds, final_true = validate(
            model, val_dataloader, device, criterion
        )
        
        class_names = ['T0', 'T2/4']
        
        cm = confusion_matrix(final_true, final_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        report = classification_report(final_true, final_preds, target_names=class_names, output_dict=True)
        with open(os.path.join(output_dir, 'classification_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nImproved Training Results:")
        print(f"Final Validation F1 Score: {final_val_f1:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"All results saved to: {output_dir}")
        
        return best_model_path
        
    except Exception as e:
        print(f"erroe during training：{e}")
        import traceback
        traceback.print_exc()
        return None

def load_and_predict(model_path, text, tokenizer_name="dmis-lab/biobert-base-cased-v1.2"):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint['model_config']
        
        model = BioBERTClassifier(
            model_config['model_name'], 
            model_config['num_classes']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=model_config['max_length'],
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_names = ['T0', 'T2/4']
        
        return {
            'predicted_class': class_names[predicted_class],
            'predicted_class_id': predicted_class,
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().numpy().tolist()
        }
    except Exception as e:
        print(f"An error occurred during prediction:{e}")
        return None
