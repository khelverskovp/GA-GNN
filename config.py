# config.py

config = {
    "experiment_number": "name_of_experiment",
    "target_index": 0,
    "max_epochs": 1000,
    "patience": 30,
    "alpha": 0.9,
    "batch_size": 100,
    "lr": 5e-4,
    "state_dim": 64,
    "num_message_passing_rounds": 4,
    "train_frac": 0.85,
    "val_frac": 0.075,
    "min_lr": 1e-6,
    "normalize_targets": False,
    "seed": 0,
    "split_sizes": [110000, 10000, 10831],
}
