import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def convert_seconds_to_datetime(seconds):
    """Convert seconds since start of year to datetime components"""
    local_time_date = pd.to_datetime(seconds, unit='s')
    return {
        'year': local_time_date.year,
        'month': local_time_date.month,
        'doy': local_time_date.dayofyear,
        'hour': local_time_date.hour,
        'minute': local_time_date.minute
    }

# Load the data
def load_data(file_path):
    """Load data from HDF5 file"""
    with h5py.File(file_path, 'r') as f:
        # Get base features
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        f107 = f['f107'][:]
        kp = f['kp'][:]
        dip = f['dip'][:]
        
        # Process temporal features
        temporal_features = []
        for seconds in local_time:
            dt = convert_seconds_to_datetime(seconds)
            temporal_features.append([
                dt['month'],
                dt['doy'],
                dt['hour']
            ])
        
        temporal_features = np.array(temporal_features)
        
        # Combine all features
        X = np.column_stack([
            latitude,
            longitude,
            temporal_features,
            f107,
            kp,
            dip
        ])
        
        # Target parameters (Chapman parameters)
        y = f['fit_results/chapman_params'][:]
        
        # Only use good fits
        is_good_fit = f['fit_results/is_good_fit'][:]
        X = X[is_good_fit]
        y = y[is_good_fit]
        
    return X, y

# Create Dataset class
class ChapmanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the MLP model
class ChapmanMLP(nn.Module):
    def __init__(self, input_size=8, hidden_sizes=[128, 64], output_size=7):
        """
        MLP model to predict Chapman function parameters
        
        Parameters:
        -----------
        input_size : int
            Number of input features:
            - latitude, longitude (2)
            - month, doy, hour (3)
            - f107, kp, dip (3)
        hidden_sizes : list
            List of hidden layer sizes
        output_size : int
            Number of output parameters (Nmax2, hmax2, H, alpha2, Nmax1, hmax1, alpha1)
        """
        super(ChapmanMLP, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    """Train the model"""
    model = model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create model-specific directory
    model_dir = './data/fit_results/filter_negative'
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,  # Save full training history
                'val_losses': val_losses,      # Save full validation history
            }, f'{model_dir}/best_chapman_mlp.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save final training history
    np.save(f'{model_dir}/train_losses.npy', train_losses)
    np.save(f'{model_dir}/val_losses.npy', val_losses)
    
    return train_losses, val_losses

def main():
    # Load and preprocess data
    X, y = load_data('./data/filtered/electron_density_profiles_2023_with_fits.h5')
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = ChapmanDataset(X_train, y_train)
    test_dataset = ChapmanDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChapmanMLP(input_size=8, hidden_sizes=[128, 64])  # Updated input size
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        num_epochs=50, device=device
    )
    
    # Save the scalers for later use
    model_dir = './data/fit_results/filter_negative'
    np.save(f'{model_dir}/X_scaler.npy', X_scaler)
    np.save(f'{model_dir}/y_scaler.npy', y_scaler)
    
    print("Training completed! Model and scalers have been saved in the filter_negative directory.")

if __name__ == "__main__":
    main() 