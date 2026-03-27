import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

from .classification_model import build_classifier
from .classification_loader import load_classification_data

# Project root is one level above this file (classification/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
CONFIG = {
    'data_root': os.path.join(_PROJECT_ROOT, 'final_dataset', 'images'),
    'model_type': 'complex', 
    'input_size': 128,        
    'batch_size': 16,         
    'learning_rate': 1e-4,    
    'epochs': 5,              
    'save_dir': os.path.join(_PROJECT_ROOT, 'checkpoint'),
    'model_name': 'breast_cancer_classifier.pth',
    'use_augmentation': True, 
    'use_balanced_sampler': True,  
}

# Quick configs for different scenarios
# CPU_FAST_CONFIG: batch_size=32, epochs=5, model_type='simple'
# GPU_FULL_CONFIG: batch_size=16, epochs=30, model_type='complex'


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        accuracy = 100 * correct / total
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc, all_preds, all_labels


def calculate_metrics(preds, labels):
    """Calculate additional metrics"""
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    # Precision, Recall, F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def train(config_overrides=None):
    runtime_config = CONFIG.copy()
    if config_overrides:
        runtime_config.update(config_overrides)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available, using CPU (training will be VERY slow)")
        print("   To use GPU:")
        print("   1. Install CUDA toolkit from NVIDIA")
        print("   2. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   3. Restart Python\n")
    else:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Load data
    print("Loading training data...")
    train_loader, train_dataset = load_classification_data(
        runtime_config['data_root'],
        batch_size=runtime_config['batch_size'],
        shuffle=not runtime_config['use_balanced_sampler'],  # sampler overrides shuffle
        augment=runtime_config['use_augmentation'],
        balance=runtime_config['use_balanced_sampler']
    )

    if len(train_dataset) == 0:
        print("No training data found!")
        return

    # Class counts for pos_weight
    n_class0 = train_dataset.labels.count(0)
    n_class1 = train_dataset.labels.count(1)
    pos_weight_value = n_class0 / max(n_class1, 1)

    print(f"\nDataset size: {len(train_dataset)}")
    print(f"  Class 0 (non-cancerous): {n_class0}")
    print(f"  Class 1 (cancerous):     {n_class1}")
    print(f"  pos_weight (class imbalance ratio): {pos_weight_value:.2f}")
    print(f"  Balanced sampler: {'ON' if runtime_config['use_balanced_sampler'] else 'OFF'}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Warn about CPU training time
    if device == "cpu" and len(train_loader) > 500:
        estimated_time_per_epoch = len(train_loader) * 0.5  # ~0.5 sec per batch on CPU
        print(f"\n💡 INFO: Training on CPU (optimized settings)")
        print(f"   Batches per epoch: {len(train_loader)}")
        print(f"   Estimated time per epoch: ~{estimated_time_per_epoch/60:.1f} minutes")
        print(f"   Total estimated time: ~{estimated_time_per_epoch * runtime_config['epochs'] / 60:.1f} minutes for {runtime_config['epochs']} epochs")
        print(f"\n   Starting training automatically...\n")
    
    print()
    
    # Initialize model
    model = build_classifier(runtime_config['model_type'], input_size=runtime_config['input_size'])
    print(f"Using model type: {runtime_config['model_type']}")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Loss and optimizer
    # pos_weight tells the loss to penalise missing a cancerous case ~3.6× more
    pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=runtime_config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Create checkpoint directory
    os.makedirs(runtime_config['save_dir'], exist_ok=True)
    
    # Training loop
    print("Starting training...\n")
    best_acc = 0.0
    current_lr = runtime_config['learning_rate']
    
    for epoch in range(runtime_config['epochs']):
        print(f"Epoch {epoch+1}/{runtime_config['epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(train_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                runtime_config['save_dir'],
                f"{runtime_config['model_type']}_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'model_type': runtime_config['model_type'],
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            best_model_path = os.path.join(runtime_config['save_dir'], runtime_config['model_name'])
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")
        
        print()
    
    print("Training completed!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved at: {os.path.join(runtime_config['save_dir'], runtime_config['model_name'])}")


if __name__ == "__main__":
    train()
