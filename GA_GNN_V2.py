import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class RBFExpansion(nn.Module):
    def __init__(self, num_rbf=20, r_cut=5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.r_cut = r_cut
        self.register_buffer('frequencies', torch.arange(1, num_rbf + 1).float() * math.pi / r_cut)

    def forward(self, distances):  
        r = distances.clamp(min=1e-6)
        rbf = torch.sin(r * self.frequencies) / r  
        return rbf

def cosine_cutoff(r, r_cut):
    cutoff = 0.5 * (torch.cos(math.pi * r / r_cut) + 1.0)
    return cutoff * (r < r_cut).float()

class GAGNN_V2_dipol(nn.Module):
    def __init__(self, output_dim=1, state_dim=128, num_message_passing_rounds=4, num_atom_types=5, num_rbf=20):
        super().__init__()
        # parameters
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.r_cut = 5.0
        self.num_atom_types = num_atom_types
        
        # atom type embeddings 
        self.atom_embedding_scalar = nn.Embedding(num_atom_types, state_dim) # scalar
        self.atom_embedding_tri = nn.Embedding(num_atom_types, state_dim)  # trivector

        # --- Message block ---
        # RBF expansion
        self.scalar_rbf_expansion = RBFExpansion(num_rbf=num_rbf, r_cut=self.r_cut)
        self.shared_rbf = nn.ModuleList([
            nn.Linear(num_rbf, 5 * state_dim)
            for _ in range(num_message_passing_rounds)
        ])

        # Shared MLP for message block
        self.shared_phi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 5 * state_dim),
                nn.SiLU(),
                nn.Linear(5 * state_dim, 5 * state_dim)
            ) for _ in range(num_message_passing_rounds)
        ])

        # --- Update block ---
        # U and V projections for state in update block
        self.linear_U = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])
        self.linear_V = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])

        # geometric product weights for update block
        self.weighted_gp_weights_Z1 = nn.ParameterList([
            nn.Parameter(torch.empty(20, state_dim).normal_(mean=0, std=1 / math.sqrt(8)))
            for _ in range(num_message_passing_rounds)
        ])

        self.weighted_gp_weights_Z2 = nn.ParameterList([
            nn.Parameter(torch.empty(20, state_dim).normal_(mean=0, std=1 / math.sqrt(8)))
            for _ in range(num_message_passing_rounds)
        ])

        # linear layers for geometric product in update block
        self.gp_linear_Z1 = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])
        self.gp_linear_Z2 = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])
        
        # MLP for update block per atom type
        self.update_mlps = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * state_dim, 4 * state_dim),
                    nn.SiLU(),
                    nn.Linear(4 * state_dim, 4 * state_dim)
                ) for _ in range(num_atom_types)
            ]) for _ in range(num_message_passing_rounds)
        ])

        # index mapping for geometric product
        self.register_buffer("gp_idx", torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 4, 14, 2, 7, 11, 5],
            [2, 12, 0, 5, 9, 3, 7, 6],
            [3, 6, 13, 0, 7, 10, 1, 4],
            [4, 10, 1, 7, 8, 14, 5, 11],
            [5, 7, 11, 2, 6, 8, 12, 9],
            [6, 3, 7, 9, 13, 4, 8, 10],
            [7, 5, 6, 4, 11, 9, 10, 8]
        ], dtype=torch.long))

        # Geometric product weights index mapping
        self.register_buffer("w_idx", torch.tensor([
            [ 0,  1,  1,  1,  2,  2,  2,  3],
            [ 4,  5,  6,  6,  7,  8,  7,  9],
            [ 4,  6,  5,  6,  7,  7,  8,  9],
            [ 4,  6,  6,  5,  8,  7,  7,  9],
            [10, 11, 11, 12, 13, 14, 14, 15],
            [10, 12, 11, 11, 14, 13, 14, 15],
            [10, 11, 12, 11, 14, 14, 13, 15],
            [16, 17, 17, 17, 18, 18, 18, 19]
        ], dtype=torch.long))

    def weighted_multivector_product(self, mv1, mv2, W):
        B, _, D = torch.broadcast_shapes(mv1.shape, mv2.shape)
        outer = mv1[:, :, None, :] * mv2[:, None, :, :] * W[self.w_idx][None]
        combined = torch.zeros(B, 16, D, device=mv1.device)
        combined.index_add_(1, self.gp_idx.flatten(), outer.view(B, 64, D))
        return combined[:, 0:8, :] - combined[:, 8:16, :]

    def forward(self, x):
        N = x['num_nodes']
        device = x['atomic_numbers'].device  

        state = torch.zeros((N, 8, self.state_dim), device=device)  
        state[:, 0] = self.atom_embedding_scalar(x['atomic_numbers'])  
        state[:, 7] = self.atom_embedding_tri(x['atomic_numbers']) 

        edge_lengths = x['edge_lengths'].unsqueeze(1)
        rbf = self.scalar_rbf_expansion(edge_lengths)
        cutoff = cosine_cutoff(edge_lengths, self.r_cut)

        # Precompute atom type masks
        atom_types = x['atomic_numbers'] 
        atom_type_masks = [(atom_types == t) for t in range(self.num_atom_types)]

        for r in range(self.num_message_passing_rounds):
            senders = x['edge_list'][:, 0]
            receivers = x['edge_list'][:, 1]
						
            # --- Message block ---
            # RBF and shared MLP
            rbf_out = self.shared_rbf[r](rbf) * cutoff 
            phi_out = self.shared_phi[r](state[senders][:, 0])
            gated = phi_out * rbf_out 
            g_s, g_v, g_d, g_b, g_t = gated.chunk(5, dim=-1) 

            # Aggregate scalar message
            state[:, 0].index_add_(0, receivers, g_s)

            # Aggregate vector message
            message_v = g_v[:, None, :] * state[senders][:, 1:4] + g_d[:, None, :] * x['edge_vectors'][:, :, None]
            state[:, 1:4].index_add_(0, receivers, message_v)

            # Aggregate bivector message
            message_b = g_b[:, None, :] * state[senders][:, 4:7]
            state[:, 4:7].index_add_(0, receivers, message_b)

            # Aggregate trivector message
            message_t = g_t * state[senders][:, 7] 
            state[:, 7].index_add_(0, receivers, message_t)

            # --- Update block ---
            # projections U and V
            U = self.linear_U[r](state)  
            V = self.linear_V[r](state)

            # first geometric product
            Z_1 = self.weighted_multivector_product(U, V, self.weighted_gp_weights_Z1[r])
            # projection for Z1
            Z1_l = self.gp_linear_Z1[r](Z_1)

            # second geometric product
            Z_2 = self.weighted_multivector_product(U, Z1_l , self.weighted_gp_weights_Z2[r])
            # projection for Z2
            Z2_l = self.gp_linear_Z2[r](Z_2)

            # norm of V[:, 1:4]
            v_norm = V[:, 1:4].norm(dim=1) 

            # update input for MLP
            update_input = torch.cat([state[:, 0], v_norm], dim=-1)

            # apply MLP and split into 4 gates
            a = torch.zeros((N, 4 * self.state_dim), device=device)
            for t in range(self.num_atom_types):
                mask = atom_type_masks[t]
                if mask.any():
                    a[mask] = self.update_mlps[r][t](update_input[mask])
            
            # Now split a into 4 gates
            a_s, a_v, a_b, a_t = a.chunk(4, dim=-1)

            # compute residual update for each component
            delta_scalar = a_s * (U[:, 0] + Z1_l[:, 0] + Z2_l[:, 0])
            delta_vector = a_v[:, None, :] * (U[:, 1:4] + Z1_l[:, 1:4] + Z2_l[:, 1:4])
            delta_bivector = a_b[:, None, :] * (U[:, 4:7] + Z1_l[:, 4:7] + Z2_l[:, 4:7])
            delta_trivector = a_t * (U[:, 7] + Z1_l[:, 7] + Z2_l[:, 7])
            
            # Apply residual update
            new_state = torch.zeros_like(state)
            new_state[:, 0] = state[:, 0] + delta_scalar
            new_state[:, 1:4] = state[:, 1:4] + delta_vector
            new_state[:, 4:7] = state[:, 4:7] + delta_bivector
            new_state[:, 7] = state[:, 7] + delta_trivector
            state = new_state

        # Output layer 
        # Scalar and vector grades
        s_i = state[:, 0]                   
        v_i = state[:, 1:4]                  

        # Reduce over state_dim to get scalar and vector
        q_i = s_i.sum(dim=-1)              
        mu_atom = v_i.sum(dim=2)            

        # Compute scalar * position
        q_pos = q_i.unsqueeze(1) * x['node_coordinates']  

        # Add vector and scalar*pos contributions
        mu_node = mu_atom + q_pos          

        # Aggregate per-graph
        dipole_vector = torch.zeros((x['num_graphs'], 3), device=device)
        dipole_vector.index_add_(0, x['node_graph_index'], mu_node)

        return dipole_vector.norm(dim=1, keepdim=True)   

class GAGNN_V2_scalarAndR2(nn.Module):
    def __init__(self, output_dim=1, state_dim=128, num_message_passing_rounds=4, num_atom_types=5, num_rbf=20):
        super().__init__()
        # parameters
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.r_cut = 5.0
        self.num_atom_types = num_atom_types
        
        # atom type embeddings 
        self.atom_embedding_scalar = nn.Embedding(num_atom_types, state_dim) # scalar
        self.atom_embedding_tri = nn.Embedding(num_atom_types, state_dim)  # trivector

        # --- Message block ---
        # RBF expansion
        self.scalar_rbf_expansion = RBFExpansion(num_rbf=num_rbf, r_cut=self.r_cut)
        self.shared_rbf = nn.ModuleList([
            nn.Linear(num_rbf, 5 * state_dim)
            for _ in range(num_message_passing_rounds)
        ])

        # Shared MLP for message block
        self.shared_phi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 5 * state_dim),
                nn.SiLU(),
                nn.Linear(5 * state_dim, 5 * state_dim)
            ) for _ in range(num_message_passing_rounds)
        ])

        # --- Update block ---
        # U and V projections for state in update block
        self.linear_U = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])
        self.linear_V = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])

        # geometric product weights for update block
        self.weighted_gp_weights_Z1 = nn.ParameterList([
            nn.Parameter(torch.empty(20, state_dim).normal_(mean=0, std=1 / math.sqrt(8)))
            for _ in range(num_message_passing_rounds)
        ])

        self.weighted_gp_weights_Z2 = nn.ParameterList([
            nn.Parameter(torch.empty(20, state_dim).normal_(mean=0, std=1 / math.sqrt(8)))
            for _ in range(num_message_passing_rounds)
        ])

        # linear layers for geometric product in update block
        self.gp_linear_Z1 = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])
        self.gp_linear_Z2 = nn.ModuleList([
            nn.Linear(state_dim, state_dim, bias=False) for _ in range(num_message_passing_rounds)
        ])
        
        # MLP for update block per atom type
        self.update_mlps = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * state_dim, 4 * state_dim),
                    nn.SiLU(),
                    nn.Linear(4 * state_dim, 4 * state_dim)
                ) for _ in range(num_atom_types)
            ]) for _ in range(num_message_passing_rounds)
        ])

        # index mapping for geometric product
        self.register_buffer("gp_idx", torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 4, 14, 2, 7, 11, 5],
            [2, 12, 0, 5, 9, 3, 7, 6],
            [3, 6, 13, 0, 7, 10, 1, 4],
            [4, 10, 1, 7, 8, 14, 5, 11],
            [5, 7, 11, 2, 6, 8, 12, 9],
            [6, 3, 7, 9, 13, 4, 8, 10],
            [7, 5, 6, 4, 11, 9, 10, 8]
        ], dtype=torch.long))

        # Geometric product weights index mapping
        self.register_buffer("w_idx", torch.tensor([
            [ 0,  1,  1,  1,  2,  2,  2,  3],
            [ 4,  5,  6,  6,  7,  8,  7,  9],
            [ 4,  6,  5,  6,  7,  7,  8,  9],
            [ 4,  6,  6,  5,  8,  7,  7,  9],
            [10, 11, 11, 12, 13, 14, 14, 15],
            [10, 12, 11, 11, 14, 13, 14, 15],
            [10, 11, 12, 11, 14, 14, 13, 15],
            [16, 17, 17, 17, 18, 18, 18, 19]
        ], dtype=torch.long))

    def weighted_multivector_product(self, mv1, mv2, W):
        B, _, D = torch.broadcast_shapes(mv1.shape, mv2.shape)
        outer = mv1[:, :, None, :] * mv2[:, None, :, :] * W[self.w_idx][None]
        combined = torch.zeros(B, 16, D, device=mv1.device)
        combined.index_add_(1, self.gp_idx.flatten(), outer.view(B, 64, D))
        return combined[:, 0:8, :] - combined[:, 8:16, :]

    def forward(self, x):
        N = x['num_nodes']
        device = x['atomic_numbers'].device  

        state = torch.zeros((N, 8, self.state_dim), device=device)  
        state[:, 0] = self.atom_embedding_scalar(x['atomic_numbers'])  
        state[:, 7] = self.atom_embedding_tri(x['atomic_numbers']) 

        edge_lengths = x['edge_lengths'].unsqueeze(1)
        rbf = self.scalar_rbf_expansion(edge_lengths)
        cutoff = cosine_cutoff(edge_lengths, self.r_cut)

        # Precompute atom type masks
        atom_types = x['atomic_numbers'] 
        atom_type_masks = [(atom_types == t) for t in range(self.num_atom_types)]

        for r in range(self.num_message_passing_rounds):
            senders = x['edge_list'][:, 0]
            receivers = x['edge_list'][:, 1]
						
            # --- Message block ---
            # RBF and shared MLP
            rbf_out = self.shared_rbf[r](rbf) * cutoff 
            phi_out = self.shared_phi[r](state[senders][:, 0])
            gated = phi_out * rbf_out 
            g_s, g_v, g_d, g_b, g_t = gated.chunk(5, dim=-1) 

            # Aggregate scalar message
            state[:, 0].index_add_(0, receivers, g_s)

            # Aggregate vector message
            message_v = g_v[:, None, :] * state[senders][:, 1:4] + g_d[:, None, :] * x['edge_vectors'][:, :, None]
            state[:, 1:4].index_add_(0, receivers, message_v)

            # Aggregate bivector message
            message_b = g_b[:, None, :] * state[senders][:, 4:7]
            state[:, 4:7].index_add_(0, receivers, message_b)

            # Aggregate trivector message
            message_t = g_t * state[senders][:, 7] 
            state[:, 7].index_add_(0, receivers, message_t)

            # --- Update block ---
            # projections U and V
            U = self.linear_U[r](state)  
            V = self.linear_V[r](state)

            # first geometric product
            Z_1 = self.weighted_multivector_product(U, V, self.weighted_gp_weights_Z1[r])
            # projection for Z1
            Z1_l = self.gp_linear_Z1[r](Z_1)

            # second geometric product
            Z_2 = self.weighted_multivector_product(U, Z1_l , self.weighted_gp_weights_Z2[r])
            # projection for Z2
            Z2_l = self.gp_linear_Z2[r](Z_2)

            # norm of V[:, 1:4]
            v_norm = V[:, 1:4].norm(dim=1) 

            # update input for MLP
            update_input = torch.cat([state[:, 0], v_norm], dim=-1)

            # apply MLP and split into 4 gates
            a = torch.zeros((N, 4 * self.state_dim), device=device)
            for t in range(self.num_atom_types):
                mask = atom_type_masks[t]
                if mask.any():
                    a[mask] = self.update_mlps[r][t](update_input[mask])
            
            # Now split a into 4 gates
            a_s, a_v, a_b, a_t = a.chunk(4, dim=-1)

            # compute residual update for each component
            delta_scalar = a_s * (U[:, 0] + Z1_l[:, 0] + Z2_l[:, 0])
            delta_vector = a_v[:, None, :] * (U[:, 1:4] + Z1_l[:, 1:4] + Z2_l[:, 1:4])
            delta_bivector = a_b[:, None, :] * (U[:, 4:7] + Z1_l[:, 4:7] + Z2_l[:, 4:7])
            delta_trivector = a_t * (U[:, 7] + Z1_l[:, 7] + Z2_l[:, 7])
            
            # Apply residual update
            new_state = torch.zeros_like(state)
            new_state[:, 0] = state[:, 0] + delta_scalar
            new_state[:, 1:4] = state[:, 1:4] + delta_vector
            new_state[:, 4:7] = state[:, 4:7] + delta_bivector
            new_state[:, 7] = state[:, 7] + delta_trivector
            state = new_state

        # Output layer
        # -- R2 prediction --
        #scalar_state = state[:, 0]  
        #atomwise_outputs = scalar_state.sum(dim=-1, keepdim=True) 

        # Compute squared distance from origin for each atom
        #r2 = x['node_coordinates'].pow(2).sum(dim=-1, keepdim=True)  

        # Multiply each atom's output with squared position norm
        #weighted_contribs = atomwise_outputs * r2  

        # Sum over atoms per graph
        #output = torch.zeros((x['num_graphs'], 1), device=scalar_state.device)
        #output.index_add_(0, x['node_graph_index'], weighted_contribs)

        #return output 

        # -- scalar prediction --
        scalar_state = state[:, 0]
        # Sum over feature dimensions
        atomwise_contributions = scalar_state.sum(dim=-1)

        # Sum atomwise predictions into per-graph outputs
        output = torch.zeros(x['num_graphs'], device=scalar_state.device)  
        output.index_add_(0, x['node_graph_index'], atomwise_contributions)

        return output.unsqueeze(-1)
