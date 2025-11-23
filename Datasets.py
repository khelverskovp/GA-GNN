import torch
import numpy as np
from torch_geometric.datasets import QM9
from torch.utils.data import Dataset

class QM93DGraphs(Dataset):
    def __init__(self, root='./data/QM9', transform=None, num_molecules=None, target_index=0):
        self.dataset = QM9(root)
        if num_molecules is not None:
            self.dataset = self.dataset[:num_molecules]
        
        self.atomic_number_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        self.transform = transform
        self.target_index = target_index
        self.ATOMIC_MASSES = {
            1: 1.0079,
            6: 12.0107,
            7: 14.0067,
            8: 15.9994,
            9: 18.9984,
            }

        # load atomref 
        self.atomref = self.dataset.atomref(self.target_index)  # tensor [max_Z+1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        pos = data.pos  # [N, 3]
        z = data.z
        y = data.y
        target_index = self.target_index
        cutoff = 5.0

        # Compute atomic masses
        masses = torch.tensor([self.ATOMIC_MASSES.get(zi.item(), 0.0) for zi in z], dtype=pos.dtype)  # [N]
        total_mass = masses.sum()
        center_of_mass = (masses[:, None] * pos).sum(dim=0) / total_mass  # [3]

        # Centralize using center of mass
        pos = pos - center_of_mass

        # Remap atomic numbers
        mapped_z = torch.tensor([self.atomic_number_to_index.get(zi.item(), 0) for zi in z])

        # Build fully connected graph with cutoff
        num_nodes = pos.size(0)
        row_idx, col_idx = torch.meshgrid(
            torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij'
        )
        mask = row_idx != col_idx
        row_idx = row_idx[mask]
        col_idx = col_idx[mask]
        edge_index = torch.stack([row_idx, col_idx], dim=1)

        edge_vectors = pos[edge_index[:, 1]] - pos[edge_index[:, 0]]
        edge_lengths = torch.norm(edge_vectors, dim=1)
        within_cutoff = edge_lengths < cutoff
        edge_index = edge_index[within_cutoff]
        edge_vectors = edge_vectors[within_cutoff]
        edge_lengths = edge_lengths[within_cutoff]
        edge_vectors = edge_vectors / (edge_lengths.unsqueeze(1))

        if self.atomref is not None:
            atomref_sum = self.atomref[z].sum()
        else:
            atomref_sum = torch.tensor(0.0)

        return {
            'node_coordinates': pos,
            'atomic_numbers': mapped_z,
            'original_atomic_numbers': z,
            'edge_list': edge_index,
            'edge_vectors': edge_vectors,
            'edge_lengths': edge_lengths,
            'label': y[0, target_index],
            'atomref': atomref_sum,
        }
