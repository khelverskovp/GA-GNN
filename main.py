import os
import sys
import signal
import json
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Datasets import QM93DGraphs
from GA_GNN import GAGNN_dipol, GAGNN_alpha, GAGNN_homo, GAGNN_R2
from config import config

# -----------------------
# Utilities
# -----------------------
def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def get_run_dir(exp_num):
    base = ensure_dir("runs")
    run_dir = ensure_dir(os.path.join(base, f"exp_{exp_num}"))
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    ensure_dir(os.path.join(run_dir, "logs"))
    ensure_dir(os.path.join(run_dir, "saved_models")) 
    return run_dir

class DotDict(dict):
    """dict with attribute access"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# -----------------------
# Target normalization
# -----------------------
class TargetNormalizer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.mean = None
        self.std = None

    def fit(self, targets):
        if self.enabled:
            self.mean = targets.mean()
            self.std = targets.std()

    def normalize(self, targets):
        if self.enabled:
            mean = self.mean
            std = self.std
            if torch.is_tensor(targets):
                if torch.is_tensor(mean): mean = mean.to(targets.device)
                if torch.is_tensor(std):  std  = std.to(targets.device)
                return (targets - mean) / (std)
            else:
                return (targets - float(mean)) / float(std)
        return targets

    def denormalize(self, targets):
        if self.enabled:
            mean = self.mean
            std = self.std
            if torch.is_tensor(targets):
                if torch.is_tensor(mean): mean = mean.to(targets.device)
                if torch.is_tensor(std):  std  = std.to(targets.device)
                return targets * (std) + mean
            else:
                return targets * float(std) + float(mean)
        return targets

    def enable(self): self.enabled = True
    def disable(self): self.enabled = False

    def state_dict(self):
        # serialize to CPU tensors or floats
        return {
            "enabled": self.enabled,
            "mean": None if self.mean is None else float(self.mean.detach().cpu()),
            "std": None if self.std is None else float(self.std.detach().cpu()),
        }

    def load_state_dict(self, state):
        self.enabled = state.get("enabled", True)
        mean = state.get("mean", None)
        std = state.get("std", None)
        self.mean = None if mean is None else torch.tensor(mean, dtype=torch.float32)
        self.std = None if std is None else torch.tensor(std, dtype=torch.float32)

# -----------------------
# Dataloader helpers
# -----------------------
def collate_graphs(batch):
    node_offset = 0
    all_coords, all_atomic_numbers, all_original_z, all_graph_indices = [], [], [], []
    all_edges, all_edge_vectors, all_edge_lengths, all_labels = [], [], [], []

    for i, graph in enumerate(batch):
        num_nodes = graph['node_coordinates'].size(0)
        all_coords.append(graph['node_coordinates'])
        all_atomic_numbers.append(graph['atomic_numbers'])
        all_original_z.append(graph['original_atomic_numbers'])
        all_graph_indices.append(torch.full((num_nodes,), i, dtype=torch.long))
        all_edges.append(graph['edge_list'] + node_offset)
        all_edge_vectors.append(graph['edge_vectors'])
        all_edge_lengths.append(graph['edge_lengths'])
        all_labels.append(graph['label'].unsqueeze(0))
        node_offset += num_nodes

    return {
        'num_nodes': node_offset,
        'num_graphs': len(batch),
        'node_coordinates': torch.cat(all_coords, dim=0),
        'atomic_numbers': torch.cat(all_atomic_numbers, dim=0),
        'original_atomic_numbers': torch.cat(all_original_z, dim=0),
        'edge_list': torch.cat(all_edges, dim=0),
        'edge_vectors': torch.cat(all_edge_vectors, dim=0),
        'edge_lengths': torch.cat(all_edge_lengths, dim=0),
        'node_graph_index': torch.cat(all_graph_indices, dim=0),
        'labels': torch.cat(all_labels, dim=0),
    }

def move_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

# -----------------------
# Training / Eval
# -----------------------
def train(model, loader, optimizer, device, epoch, normalizer, cfg):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch['labels'].unsqueeze(1)
        target = normalizer.normalize(target)
        if cfg.model_name == "alpha":
            loss = F.l1_loss(pred, target)  # MAE loss for alpha
        else:
            loss = F.mse_loss(pred, target)  # MSE loss otherwise
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, normalizer, cfg):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            pred = model(batch)
            target = batch['labels'].unsqueeze(1)
            target = normalizer.normalize(target)
            if cfg.model_name == "alpha":
                total_loss += F.l1_loss(pred, target).item()
            else:
                total_loss += F.mse_loss(pred, target).item()
    return total_loss / len(loader)

# -----------------------
# Checkpointing
# -----------------------
def rng_state_state_dict():
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }

def load_rng_state(state):
    if state is None: return
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])

def save_checkpoint(run_dir, state, is_best=False, tag="last"):
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{tag}.ckpt")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(run_dir, "checkpoints", "best.ckpt")
        torch.save(state, best_path)
    return ckpt_path

def try_load_checkpoint(run_dir, tag="last"):
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{tag}.ckpt")
    if os.path.isfile(ckpt_path):
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return None

# -----------------------
# Main
# -----------------------
def main():
    # Init W&B run with resume support
    cfg = DotDict(config)
    exp_num = cfg["experiment_number"]
    run_dir = get_run_dir(exp_num)

    # Save config snapshot to run_dir
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- Seeds: fix training randomness, vary only the data split ----
    train_seed = int(getattr(cfg, "train_seed", 0))              # fixed across runs (can set in config)
    split_seed = int(getattr(cfg, "split_seed", getattr(cfg, "seed", 0)))  # varies across runs

    set_seed(train_seed)  # fixes model init, dropout, etc.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = QM93DGraphs(target_index=cfg.target_index)

    # ---- Build/restore dataset splits (driven ONLY by split_seed) ----
    splits_path = os.path.join(run_dir, "splits.pkl")
    splits = None
    if os.path.isfile(splits_path):
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        if int(splits.get("split_seed", -1)) != split_seed:
            print(f"[INFO] Existing splits were made with split_seed={splits.get('split_seed')}, "
                f"but cfg split_seed={split_seed}. Regenerating splits.")
            splits = None

    if splits is None:
        train_len, val_len, test_len = cfg.split_sizes
        assert len(dataset) == sum(cfg.split_sizes), "Dataset size mismatch!"
        split_gen = torch.Generator().manual_seed(split_seed)
        tr_sub, va_sub, te_sub = random_split(dataset, [train_len, val_len, test_len], generator=split_gen)
        train_indices, val_indices, test_indices = tr_sub.indices, va_sub.indices, te_sub.indices
        with open(splits_path, "wb") as f:
            pickle.dump({
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
                "split_seed": split_seed
            }, f)
    else:
        train_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]


    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_graphs, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_graphs, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_graphs, drop_last=True)

    # Normalizer
    normalizer = TargetNormalizer(enabled=cfg.normalize_targets)
    # Fit once on training set (labels) if fresh run; otherwise load from checkpoint
    # (We still compute here for safety if no checkpoint is found)
    def fit_normalizer_from_train():
        if cfg.normalize_targets:
            all_targets = []
            for idx in train_indices:
                d = dataset[idx]
                all_targets.append(d['label'].unsqueeze(0))
            target_tensor = torch.cat(all_targets, dim=0).to(device)
            normalizer.fit(target_tensor)

    # Model/optim/scheduler
    # --- Model builder driven by cfg.model_name ---
    MODEL_REGISTRY = {
        "dipole": GAGNN_dipol,   # mu
        "r2":     GAGNN_R2,      # R^2
        "alpha":  GAGNN_alpha,   # polarizability
        "homo":   GAGNN_homo,    # ε_HOMO
    }

    mn = str(cfg.model_name).lower()
    if mn not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name='{cfg.model_name}'. "
                        f"Choose one of: {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[mn]
    model = ModelClass(
        output_dim=1,
        state_dim=cfg.state_dim,
        num_message_passing_rounds=cfg.num_message_passing_rounds,
        num_atom_types=5,
        num_rbf=20,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        threshold=1e-6, threshold_mode='rel',
        cooldown=2, min_lr=cfg.min_lr
    )

    # Resume if possible
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    smoothed_val_loss = None
    train_losses, val_losses = [], []

    ckpt = try_load_checkpoint(run_dir, tag="last")
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        plateau_scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt["patience_counter"]
        smoothed_val_loss = ckpt["smoothed_val_loss"]
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        normalizer.load_state_dict(ckpt["normalizer"])
        load_rng_state(ckpt.get("rng_state", None))
        print(f"Resumed from epoch {ckpt['epoch']} (best_val_loss={best_val_loss:.6f})")
        print("="*60)
        print(f" Resuming training from epoch {ckpt['epoch']} "
            f"(best_val_loss={best_val_loss:.6f}, patience_counter={patience_counter})")
        print("="*60)
    else:
        print("="*60)
        print(" Starting fresh training run")
        print("="*60)

        fit_normalizer_from_train()

    # Paths for logs / results
    loss_log_path = os.path.join(run_dir, "logs", f"train_val_losses.pkl")
    test_results_path = os.path.join(run_dir, "logs", f"test_results.pkl")
    best_model_path = os.path.join(run_dir, "saved_models", "best_model_state_dict.pt")

    # interrupter to snapshot on SIGTERM/SIGINT (e.g., preemption/timeout)
    def _snapshot_and_exit(signum, frame):
        state = {
            "epoch": max(start_epoch - 1, 1),  # last finished epoch
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": plateau_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "smoothed_val_loss": smoothed_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "normalizer": normalizer.state_dict(),
            "rng_state": rng_state_state_dict(),
            "config": dict(cfg),
            "splits": {"train": train_indices, "val": val_indices, "test": test_indices},
        }
        path = save_checkpoint(run_dir, state, is_best=False, tag="emergency")
        print(f"\nReceived signal {signum}. Saved emergency checkpoint to {path}. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _snapshot_and_exit)
    signal.signal(signal.SIGINT, _snapshot_and_exit)

    # Training loop
    max_epochs = cfg.max_epochs
    alpha = cfg.alpha
    patience = cfg.patience

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch, normalizer, cfg)
        val_loss = evaluate(model, val_loader, device, normalizer, cfg)
        smoothed_val_loss = val_loss if smoothed_val_loss is None else alpha * smoothed_val_loss + (1 - alpha) * val_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Smoothed Val Loss: {smoothed_val_loss:.4f}")

        plateau_scheduler.step(smoothed_val_loss)

        # Save latest checkpoint 
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": plateau_scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "smoothed_val_loss": smoothed_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "normalizer": normalizer.state_dict(),
            "rng_state": rng_state_state_dict(),
            "config": dict(cfg),
            "splits": {"train": train_indices, "val": val_indices, "test": test_indices},
        }
        save_checkpoint(run_dir, state, is_best=False, tag="last")

        # Track best model
        improved = smoothed_val_loss < best_val_loss
        if improved:
            best_val_loss = smoothed_val_loss
            patience_counter = 0
            save_checkpoint(run_dir, state, is_best=True, tag="last")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Persist loss curves
    with open(loss_log_path, "wb") as f:
        pickle.dump({"train": train_losses, "val": val_losses}, f)

    # Evaluate on test set using best checkpoint
    def evaluate_on_test_set():
        best_ckpt = try_load_checkpoint(run_dir, tag="best") or try_load_checkpoint(run_dir, tag="last")
        if best_ckpt is not None:
            model.load_state_dict(best_ckpt["model_state"])
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = move_to_device(batch, device)
                preds = model(batch).squeeze(1)
                preds = normalizer.denormalize(preds)
                targets = batch['labels']
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
        with open(test_results_path, "wb") as f:
            pickle.dump({
                "predictions": torch.cat(all_preds).numpy(),
                "targets": torch.cat(all_targets).numpy()
            }, f)

    evaluate_on_test_set()

if __name__ == "__main__":
    main()



