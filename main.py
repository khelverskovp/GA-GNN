import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from Datasets import QM93DGraphs
from GA_GNN_V1 import GAGNN_V2_dipol
import os
import pickle
import wandb
from config import config
import random
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# target normalization
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
            return (targets - self.mean) / (self.std)
        return targets

    def denormalize(self, targets):
        if self.enabled:
            return targets * (self.std) + self.mean
        return targets

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

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

def train(model, loader, optimizer, device, epoch, normalizer):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        pred = model(batch)
        target = batch['labels'].unsqueeze(1)
        target = normalizer.normalize(target)
        loss = F.mse_loss(pred, target) # MSE loss for all other targets
        #loss = F.l1_loss(pred, target) # MAE loss for isotropic polarizability 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if wandb.run:
            wandb.log({"batch_loss": loss.item(), "epoch": epoch})

    return total_loss / len(loader)

def evaluate(model, loader, device, normalizer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            pred = model(batch)
            target = batch['labels'].unsqueeze(1)
            target = normalizer.normalize(target)
            total_loss += F.mse_loss(pred, target).item() # MSE for all other targets
            #total_loss += F.l1_loss(pred, target, reduction='mean').item() # MAE for isotropic polarizability
    return total_loss / len(loader)

def main():
    wandb.init(project="gagnn", config=config, name="dipol_variant_4")
    cfg = wandb.config
    set_seed(cfg.seed)
    experiment_number = cfg.experiment_number
    #torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = QM93DGraphs(target_index=cfg.target_index)

    # Normalizer
    normalizer = TargetNormalizer(enabled=cfg.normalize_targets)

    train_len, val_len, test_len = cfg.split_sizes
    assert len(dataset) == sum(cfg.split_sizes), "Dataset size mismatch!"
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_graphs, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_graphs, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_graphs, drop_last=True)

    model = GAGNN_V2_dipol(output_dim=1, state_dim=cfg.state_dim, num_message_passing_rounds=cfg.num_message_passing_rounds, num_atom_types=5, num_rbf=20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        threshold=1e-6, threshold_mode='rel',
        cooldown=2, min_lr=cfg.min_lr
    )

    model_path = f"saved_models/gagnn_{experiment_number}.pt"
    loss_log_path = f"logs/train_val_losses_{experiment_number}.pkl"
    test_results_path = f"logs/test_results_{experiment_number}.pkl"
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Fit normalizer on training set 
    if cfg.normalize_targets:
        all_targets = [d['label'].unsqueeze(0) for d in train_set]
        target_tensor = torch.cat(all_targets, dim=0).to(device) 
        normalizer.fit(target_tensor)

    best_val_loss = float('inf')
    patience_counter = 0
    smoothed_val_loss = None
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch, normalizer)
        val_loss = evaluate(model, val_loader, device, normalizer)
        smoothed_val_loss = val_loss if smoothed_val_loss is None else cfg.alpha * smoothed_val_loss + (1 - cfg.alpha) * val_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Smoothed Val Loss: {smoothed_val_loss:.4f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "smoothed_val_loss": smoothed_val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })

        plateau_scheduler.step(smoothed_val_loss)

        if smoothed_val_loss < best_val_loss:
            best_val_loss = smoothed_val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                

    with open(loss_log_path, "wb") as f:
        pickle.dump({"train": train_losses, "val": val_losses}, f)

    # Evaluate on test set
    def evaluate_on_test_set():
        model.load_state_dict(torch.load(model_path))
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
            pickle.dump({"predictions": torch.cat(all_preds).numpy(), "targets": torch.cat(all_targets).numpy()}, f)

    evaluate_on_test_set()

if __name__ == "__main__":
    main()
