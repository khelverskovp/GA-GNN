# config.py

config = {
    # Directory setup
    "experiment_number": "1",  # Used to create runs/exp_<experiment_number>/ for logs & checkpoints

    # Task setup
    "target_index": 0,         # Which QM9 property to predict:
                               # 0 = dipole moment (mu), 1 = isotrpoic polarizability (alpha),
                               # 2 = HOMO energy, etc. 

    # Training schedule
    "max_epochs": 1000,        # Maximum number of training epochs
    "patience": 30,            # Early stopping patience (epochs without improvement)
    "alpha": 0.9,              # Smoothing factor for exponential moving average of val loss

    # Optimizer / learning rate
    "batch_size": 100,         # Number of graphs per training batch
    "lr": 5e-4,                # Initial learning rate
    "min_lr": 1e-6,            # Lower bound for learning rate (scheduler won’t reduce further)

    # Model architecture
    "state_dim": 128,           # Dimension of the multivector hidden state
    "num_message_passing_rounds": 4,  # Number of GNN message passing iterations

    # Data preprocessing
    "normalize_targets": False, # Normalize target values?
                               #   True  → for targets [1,2,3,4] 
                               #   False → for targets [0,5,6,7,8,9,10,11]

    # Random seeds
    "train_seed": 0,           # Global seed for weight initialization etc. (fixed across runs)
    "split_seed": 1,           # Seed controlling dataset split (train/val/test)

    # Dataset splits
    "split_sizes": [110000, 10000, 10831],  # Number of samples in [train, val, test]

    # Model choice
    "model_name": "dipole",    # Select which model to use:
                               # "dipole", "r2", "alpha", or "scalar1" for scalar property without output MLP and "scalar2" for scalar property with output MLP.

    # Whether to use W&B
    "use_wandb" : False,
}




