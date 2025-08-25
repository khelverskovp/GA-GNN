# config.py

config = {
    "experiment_number": "1", # for runs/experiment_number
    "run_name": "dipole_1", # wandb run name
    "target_index": 5,  # index 0 = mu, 1 = alpha, 2 = homo, 5 = R2
    "max_epochs": 1000,
    "patience": 30,
    "alpha": 0.9,
    "batch_size": 100,
    "lr": 5e-4,
    "state_dim": 64,
    "num_message_passing_rounds": 4,
    "min_lr": 1e-6,
    "normalize_targets": False, # True for alpha and homo. False for dipole and R^2
    "train_seed": 0,   # fixed across runs
    "split_seed": 3,   # controls the random data split
    "split_sizes": [110000, 10000, 10831],
    "model_name": "r2_gw",   # one of: "dipole", "r2", "alpha", "homo"
}

