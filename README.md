# GA-GNN: Geometric Algebra Graph Neural Network

This repository contains the implementation of GA-GNN models for "Design and Evaluation of a Geometric Algebra-Based Graph Neural Network for Molecular Property Prediction". 

## Files

- `GA_GNN`: Code for the models used in the main evaluation for each target.
- `Datasets.py`: Dataset class for QM9 with atomic numbers, positions, and edge vectors.
- `main.py`: Training and evaluation script.
- `config.py`: Configuration file for model and training parameters.
- `requirements.txt`: Python dependencies.

## Running experiments

All experiment settings are defined in **`config.py`**.  
Edit the file to choose the property you want to predict and adjust training parameters as needed.

### Key fields

- **`experiment_number`**: creates `runs/exp_<n>/` for logs and checkpoints  
- **`target_index` + `model_name`** (must match):
  - `0` + `"dipole"` → dipole moment (μ)  
  - `1` + `"alpha"`  → polarizability (α)  
  - `2` + `"homo"`   → HOMO energy (ε_HOMO)  
  - `5` + `"r2"`     → R²
- **Training hyperparameters**: `max_epochs`, `batch_size`, `lr`, `min_lr`, `patience`, `alpha`  
- **Model parameters**: `state_dim`, `num_message_passing_rounds`  
- **Data options**: `normalize_targets`, `split_seed`, `split_sizes`

### Run training

```bash
python main.py

This will create a directory `runs/exp_<experiment_number>/` containing:

- **checkpoints/**
  - `last.ckpt` — checkpoint from the last epoch  
  - `best.ckpt` — checkpoint with the best validation score  

- **logs/**
  - `train_val_losses.pkl` — training and validation loss history  
  - `test_results.pkl` — predictions and ground truth on the test set  

- **saved_models/**
  - `best_model_state_dict.pt` — best model weights (ready to load for inference)  

- `config.json` — snapshot of the configuration used for this run  
- `splits.pkl` — indices for train/validation/test splits (so splits are reproducible)  

Re-running with the same `experiment_number` resumes automatically from the last checkpoint.

