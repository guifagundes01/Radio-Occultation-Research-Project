import torch
import torch.nn as nn
import torch.optim as optim
from model.chapman_function import chapmanF2F1_torch

class ChapmanPINN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))  # Output: electron density
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_density(self, features):
        return self.forward(features)

# Physics loss: residual of the Chapman function (dN/dz - Chapman ODE RHS)
def chapman_physics_residual(model, features, chapman_params):
    # features: [batch, 12] (last column is altitude)
    features = features.clone().detach().requires_grad_(True)
    N_pred = model(features)
    altitude = features[:, -1].unsqueeze(1)
    # Unpack Chapman parameters for each sample in the batch
    Nmax, hmax, H, a1, Nm, a2, hm = [p.unsqueeze(1) for p in chapman_params]
    N_chapman = chapmanF2F1_torch(altitude, Nmax, hmax, H, a1, Nm, a2, hm)
    # Compute dN_pred/dz
    dN_dz = torch.autograd.grad(N_pred, altitude, grad_outputs=torch.ones_like(N_pred), create_graph=True)[0]
    # Compute dN_chapman/dz analytically (optional: can use autograd as well)
    dN_chapman_dz = torch.autograd.grad(N_chapman, altitude, grad_outputs=torch.ones_like(N_chapman), create_graph=True)[0]
    # Residual: difference between predicted and Chapman derivative
    residual = dN_dz - dN_chapman_dz
    return residual

# Skeleton for training loop
def train_pinn(model, optimizer, data_loader, chapman_params_loader, lambda_phys=1.0, device='cpu'):
    model.train()
    mse_loss = nn.MSELoss()
    for batch_idx, (features, target_density) in enumerate(data_loader):
        features = features.to(device)
        target_density = target_density.to(device)
        chapman_params = next(iter(chapman_params_loader))  # Should match batch
        chapman_params = [p.to(device) for p in chapman_params]
        # Data loss
        pred_density = model(features)
        data_loss = mse_loss(pred_density, target_density)
        # Physics loss
        residual = chapman_physics_residual(model, features, chapman_params)
        physics_loss = torch.mean(residual ** 2)
        # Total loss
        loss = data_loss + lambda_phys * physics_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Data Loss = {data_loss.item():.4e}, Physics Loss = {physics_loss.item():.4e}") 