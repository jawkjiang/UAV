"""Training utilities for the MATLAB-based UAV spoofing detection pipeline."""
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def _checkpoint_path(model_name: str, seed: int) -> str:
    ckpt_dir = os.path.join(config.OUTPUT_DIR, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f'{model_name}_seed{seed}.pt')


def train_one_model(model_class, model_name: str,
                    X_tr: np.ndarray, y_tr: np.ndarray,
                    X_va: np.ndarray, y_va: np.ndarray,
                    seed: int, n_features: int = 9, window_size: int = 50,
                    device: torch.device = None) -> dict:
    """Train one model with one seed.

    Models are expected to output probabilities in [0,1] (sigmoid already
    applied inside the model). BCELoss is used directly on model output.

    Returns dict with 'model' (on CPU), 'val_loss_history', 'best_epoch', 'seed'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _set_seed(seed)
    model = model_class(n_features=n_features, window_size=window_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()

    tr_loader = _make_loader(X_tr, y_tr, shuffle=True)
    va_loader = _make_loader(X_va, y_va, shuffle=False)

    best_val_loss = math.inf
    patience_counter = 0
    best_epoch = 1
    val_loss_history = []
    ckpt_path = _checkpoint_path(model_name, seed)

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        tr_loss_sum = 0.0
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            # Model already outputs sigmoid probabilities
            preds = model(X_b).squeeze(-1)
            loss = criterion(preds, y_b)
            loss.backward()
            optimiser.step()
            tr_loss_sum += loss.item() * len(y_b)
        tr_loss = tr_loss_sum / len(y_tr)

        model.eval()
        va_loss_sum = 0.0
        with torch.no_grad():
            for X_b, y_b in va_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b).squeeze(-1)
                va_loss_sum += criterion(preds, y_b).item() * len(y_b)
        va_loss = va_loss_sum / len(y_va)
        val_loss_history.append(va_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f'  [{model_name} s={seed}] ep{epoch:>3}  '
                  f'tr={tr_loss:.4f}  va={va_loss:.4f}')

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f'  [{model_name} s={seed}] early stop ep{epoch} '
                      f'(best ep{best_epoch} va={best_val_loss:.4f})')
                break

    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    model.to('cpu')
    return {'model': model, 'val_loss_history': val_loss_history,
            'best_epoch': best_epoch, 'seed': seed}


def train_multi_seed(model_class, model_name: str,
                     X_tr: np.ndarray, y_tr: np.ndarray,
                     X_va: np.ndarray, y_va: np.ndarray,
                     seeds=config.SEEDS, n_features: int = 9,
                     window_size: int = 50) -> list:
    """Train model_class with multiple seeds. Returns list of result dicts."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}\nTraining {model_name} on {device} x {len(seeds)} seeds\n{"="*60}')
    results = []
    for seed in seeds:
        print(f'\n-- seed {seed} --')
        results.append(train_one_model(
            model_class, model_name, X_tr, y_tr, X_va, y_va,
            seed=seed, n_features=n_features, window_size=window_size, device=device))
    bests = ', '.join(f's{r["seed"]}={min(r["val_loss_history"]):.4f}' for r in results)
    print(f'\n[{model_name}] done. best val losses: {bests}')
    return results
