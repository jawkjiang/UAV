"""
Training Module

Train GPS spoofing detection model with weighted loss.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
import config


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance.
    """
    
    def __init__(self, pos_weight=1.0):
        """
        Initialize loss.
        
        Args:
            pos_weight: Weight for positive class
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, outputs, targets):
        """
        Compute weighted BCE loss.
        
        Args:
            outputs: Model predictions [batch_size, 1]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Loss value
        """
        outputs = outputs.squeeze()
        
        # BCE with logit is more stable, but we use sigmoid in model
        # So we use standard BCE
        bce = -(targets * torch.log(outputs + 1e-7) + 
                (1 - targets) * torch.log(1 - outputs + 1e-7))
        
        # Apply weight to positive samples
        weights = torch.where(targets == 1, 
                             torch.tensor(self.pos_weight, device=targets.device),
                             torch.tensor(1.0, device=targets.device))
        
        weighted_bce = bce * weights
        
        return weighted_bce.mean()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
    
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to use
    
    Returns:
        Tuple of (avg_loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    return avg_loss, predictions, targets


def compute_pr_auc(targets, predictions):
    """
    Compute Precision-Recall AUC.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted probabilities
    
    Returns:
        PR-AUC score
    """
    precision, recall, _ = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall, precision)
    return pr_auc


def train_model(model, train_loader, val_loader, 
                pos_weight=1.0,
                num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                patience=config.EARLY_STOPPING_PATIENCE,
                device='cuda',
                save_path=None):
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        pos_weight: Weight for positive class
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Device to use
        save_path: Path to save best model
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pr_auc': []
    }
    
    best_val_pr_auc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        val_pr_auc = compute_pr_auc(val_targets, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pr_auc'].append(val_pr_auc)
        
        # Learning rate scheduling
        scheduler.step(val_pr_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val PR-AUC={val_pr_auc:.4f}")
        
        # Early stopping
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            
            # Save best model
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_pr_auc': val_pr_auc,
                }, save_path)
                print(f"  Saved best model (PR-AUC={val_pr_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\nTraining completed. Best Val PR-AUC: {best_val_pr_auc:.4f}")
    
    return history


if __name__ == '__main__':
    import pickle
    import json
    from model import GPSSpoofingDetector
    from window_creation import WindowDataset
    
    print("="*80)
    print("Model Training")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load windows data
    print("\n[1] Loading windows data...")
    with open(os.path.join(config.OUTPUT_DIR, 'windows_data.pkl'), 'rb') as f:
        windows_data = pickle.load(f)
    
    train_windows = windows_data['train']['windows']
    train_labels = windows_data['train']['labels']
    val_windows = windows_data['val']['windows']
    val_labels = windows_data['val']['labels']
    
    print(f"    Train: {len(train_labels)} windows")
    print(f"    Val:   {len(val_labels)} windows")
    
    # Create datasets
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Compute positive weight
    pos_weight = train_dataset.get_positive_weight()
    print(f"\n[2] Positive class weight: {pos_weight:.2f}")
    
    # Create model
    print("\n[3] Creating model...")
    n_features = train_windows.shape[2]
    model = GPSSpoofingDetector(n_features=n_features)
    model = model.to(device)
    
    print(f"    Model created with n_features={n_features}")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n[4] Training model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        pos_weight=pos_weight,
        patience=config.EARLY_STOPPING_PATIENCE,
        save_path=os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    )
    
    # Save training history
    print("\n[5] Saving training history...")
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"    ✓ Saved to {config.OUTPUT_DIR}/training_history.json")
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
