import math
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, features: int, dim: int = 4000):
        super(Encoder, self).__init__()
        self.dim = dim
        self.features = features
        self.basis = nn.Parameter(torch.randn(self.dim, self.features))  # Random basis (dim, features)
        self.base = nn.Parameter(torch.empty(self.dim).uniform_(0.0, 2 * math.pi))  # Random base (dim,)

    def encode_value(self, x):
        """Nonlinear encoding for single values"""
        x_reshaped = x.view(-1, self.features)
        temp = x_reshaped @ self.basis.T  # (batch_size, timepoints, features) @ (features, dim)
        return torch.sin(temp) * torch.cos(temp + self.base)  # (batch_size, timepoints, dim)

    def forward(self, x: torch.Tensor): 
        batch_size, n_channels, n_timepoints = x.shape
        bsize = 32  # Efficient batch size
        h = torch.empty(batch_size, self.dim, device=x.device, dtype=x.dtype)

        # print(f'bsize is {bsize} and batch_size is {batch_size}')
        for i in range(0, batch_size, bsize):
            batch = x[i:i+bsize]  # (bsize, channels, timepoints)
            batch = torch.round(batch * 2) / 2
            bsize = batch.shape[0]
            
            batch_encoded_values = self.encode_value(batch)  # Already in (bsize, timepoints, dim)
            batch_encoded_values = batch_encoded_values.view(bsize, n_timepoints, self.dim)
            permuted_sum = batch_encoded_values.sum(dim=1)

            # shifts = torch.arange(n_timepoints, device=x.device)  # [0, 1, 2, ..., n_timepoints-1]

            # permuted = torch.stack([torch.roll(batch_encoded_values[:, t, :], shifts[t].item(), dims=1) for t in range(n_timepoints)], dim=1)
            # # Sum over the timepoints dimension
            # permuted_sum = permuted.sum(dim=1)
            h[i:i+bsize] = permuted_sum  # Sum over timepoints (bundling)

        return h

    def basis_orthogonality(self):
        """Check orthogonality of basis vectors"""
        dot = torch.mm(self.basis, self.basis.T).abs()
        norm = torch.norm(self.basis, dim=1).unsqueeze(1)
        sim_mat = dot / (norm * norm.T)
        sim_mat.fill_diagonal_(0)
        return sim_mat