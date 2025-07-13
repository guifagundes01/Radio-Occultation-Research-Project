import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime, timedelta
import pandas as pd
from model.chapman_function import generate_chapman_profiles
import matplotlib.pyplot as plt
import json

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

def circular_encode(value, max_value):
    """Apply circular encoding to a value"""
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val

class ChapmanPredictor(nn.Module):
    """Stage 1: Predicts Chapman function parameters"""
    def __init__(self, input_size=11, hidden_sizes=[128, 64], output_size=7):
        super(ChapmanPredictor, self).__init__()
        
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

class ProfileCorrector(nn.Module):
    """Stage 2: Corrects the Chapman profile predictions"""
    def __init__(self, profile_size, feature_size, hidden_sizes=[128, 64]):
        super(ProfileCorrector, self).__init__()
        
        # Combined input size (features + Chapman parameters)
        combined_size = feature_size + 7  # 7 Chapman parameters
        
        # Dense layers
        layers = []
        prev_size = combined_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer (correction factors for each altitude point)
        layers.append(nn.Linear(prev_size, profile_size))
        layers.append(nn.Tanh())  # Bound corrections to [-1, 1]
        
        self.dense = nn.Sequential(*layers)
        
        # Initialize weights with smaller values
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Reduced gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features, chapman_params):
        # Combine features and Chapman parameters
        combined = torch.cat([features, chapman_params], dim=1)
        
        # Generate corrections
        corrections = self.dense(combined)
        
        return corrections

class HybridDataset(Dataset):
    """Dataset for the hybrid model"""
    def __init__(self, X, y, profiles, altitude):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.profiles = torch.FloatTensor(profiles)
        self.altitude = torch.FloatTensor(altitude)  # This is a single array of height points
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return the same altitude array for all profiles
        return self.X[idx], self.y[idx], self.profiles[idx], self.altitude

def save_training_history(train_losses, val_losses, stage1_losses, stage2_losses, model_dir):
    """Save training history to files"""
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'stage1_losses': stage1_losses,
        'stage2_losses': stage2_losses
    }
    
    # Save as numpy arrays
    np.save(f'{model_dir}/train_losses.npy', train_losses)
    np.save(f'{model_dir}/val_losses.npy', val_losses)
    np.save(f'{model_dir}/stage1_losses.npy', stage1_losses)
    np.save(f'{model_dir}/stage2_losses.npy', stage2_losses)
    
    # Save as JSON for easy reading
    with open(f'{model_dir}/training_history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=4)

def plot_training_history(train_losses, val_losses, stage1_losses, stage2_losses, model_dir):
    """Plot training history"""
    plt.figure(figsize=(15, 10))
    
    # Plot total losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot stage 1 losses
    plt.subplot(2, 2, 2)
    plt.plot(stage1_losses, label='Stage 1')
    plt.title('Stage 1 Loss (Chapman Parameters)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot stage 2 losses
    plt.subplot(2, 2, 3)
    plt.plot(stage2_losses, label='Stage 2')
    plt.title('Stage 2 Loss (Profile Corrections)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot loss components
    plt.subplot(2, 2, 4)
    plt.plot(stage1_losses, label='Stage 1')
    plt.plot(stage2_losses, label='Stage 2')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_dir}/training_history.png')
    plt.close()

def evaluate_and_plot_test_results(model_stage1, model_stage2, test_loader, device, model_dir):
    """Evaluate model performance on test set and create visualization plots"""
    model_stage1.eval()
    model_stage2.eval()
    
    # Lists to store results
    all_true_profiles = []
    all_pred_profiles = []
    all_altitudes = []
    
    with torch.no_grad():
        for X_batch, y_batch, profiles_batch, altitude_batch in test_loader:
            X_batch = X_batch.to(device)
            profiles_batch = profiles_batch.to(device)
            altitude_batch = altitude_batch.to(device)
            
            # Get predictions
            chapman_params = model_stage1(X_batch)
            chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
            corrections = model_stage2(X_batch, chapman_params)
            
            # Scale corrections
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            
            # Get final predictions
            predicted_profiles = chapman_profiles + corrections
            
            # Store results
            all_true_profiles.append(profiles_batch.cpu().numpy())
            all_pred_profiles.append(predicted_profiles.cpu().numpy())
            all_altitudes.append(altitude_batch.cpu().numpy())
    
    # Convert to numpy arrays
    all_true_profiles = np.concatenate(all_true_profiles, axis=0)
    all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
    all_altitudes = np.concatenate(all_altitudes, axis=0)
    
    # Calculate error metrics
    mse = np.mean((all_true_profiles - all_pred_profiles) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_true_profiles - all_pred_profiles))
    
    # Calculate relative error
    rel_error = np.abs(all_true_profiles - all_pred_profiles) / (np.abs(all_true_profiles) + 1e-8)
    mean_rel_error = np.mean(rel_error)
    median_rel_error = np.median(rel_error)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Sample profiles
    plt.subplot(2, 2, 1)
    sample_idx = np.random.choice(len(all_true_profiles), 5, replace=False)
    for idx in sample_idx:
        plt.semilogy(all_altitudes[idx], all_true_profiles[idx], 'b-', alpha=0.5, label='True' if idx == sample_idx[0] else None)
        plt.semilogy(all_altitudes[idx], all_pred_profiles[idx], 'r--', alpha=0.5, label='Predicted' if idx == sample_idx[0] else None)
    plt.xlabel('Altitude (km)')
    plt.ylabel('Electron Density (cm⁻³)')
    plt.title('Sample Profiles')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Error distribution
    plt.subplot(2, 2, 2)
    plt.hist(rel_error.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    # Plot 3: Mean profile comparison
    plt.subplot(2, 2, 3)
    mean_true = np.mean(all_true_profiles, axis=0)
    mean_pred = np.mean(all_pred_profiles, axis=0)
    std_true = np.std(all_true_profiles, axis=0)
    std_pred = np.std(all_pred_profiles, axis=0)
    
    plt.semilogy(all_altitudes[0], mean_true, 'b-', label='True Mean')
    plt.semilogy(all_altitudes[0], mean_pred, 'r--', label='Predicted Mean')
    plt.fill_between(all_altitudes[0], 
                     mean_true - std_true, 
                     mean_true + std_true, 
                     alpha=0.2, color='blue')
    plt.fill_between(all_altitudes[0], 
                     mean_pred - std_pred, 
                     mean_pred + std_pred, 
                     alpha=0.2, color='red')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Electron Density (cm⁻³)')
    plt.title('Mean Profile Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Scatter plot of true vs predicted
    plt.subplot(2, 2, 4)
    plt.loglog(all_true_profiles.flatten(), all_pred_profiles.flatten(), 'k.', alpha=0.1)
    plt.loglog([1e9, 1e13], [1e9, 1e13], 'r--')  # Perfect prediction line
    plt.xlabel('True Electron Density (cm⁻³)')
    plt.ylabel('Predicted Electron Density (cm⁻³)')
    plt.title('True vs Predicted')
    plt.grid(True)
    
    # Add metrics text
    metrics_text = f'MSE: {mse:.2e}\nRMSE: {rmse:.2e}\nMAE: {mae:.2e}\nMean Rel Error: {mean_rel_error:.2%}\nMedian Rel Error: {median_rel_error:.2%}'
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{model_dir}/test_results.png')
    plt.close()
    
    # Save metrics to file
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mean_rel_error': float(mean_rel_error),
        'median_rel_error': float(median_rel_error)
    }
    
    with open(f'{model_dir}/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def train_hybrid_model_end_to_end(model_stage1, model_stage2, train_loader, val_loader, 
                                 criterion, optimizer, num_epochs=30, device='cuda'):
    """Train both stages of the hybrid model with end-to-end training"""
    model_stage1 = model_stage1.to(device)
    model_stage2 = model_stage2.to(device)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    stage1_losses = []
    stage2_losses = []
    
    # Create model-specific directory
    model_dir = './data/fit_results/hybrid_model_v3'
    os.makedirs(model_dir, exist_ok=True)
    
    # Gradient clipping value
    max_grad_norm = 0.1
    
    # Loss scaling factors for end-to-end training
    stage1_scale = 0.3  # Reduced weight for parameter loss
    stage2_scale = 1.0  # Full weight for profile loss
    
    def robust_mse_loss(pred, target):
        """Calculate MSE loss with numerical stability and outlier handling"""
        # Calculate relative error
        rel_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
        
        # Clip relative error to prevent extreme values
        rel_error = torch.clamp(rel_error, 0, 10.0)
        
        # Calculate mean squared relative error
        mse = torch.mean(rel_error ** 2)
        return mse
    
    for epoch in range(num_epochs):
        # Training phase
        model_stage1.train()
        model_stage2.train()
        train_loss = 0
        stage1_loss_sum = 0
        stage2_loss_sum = 0
        batch_count = 0
        
        for X_batch, y_batch, profiles_batch, altitude_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            profiles_batch = profiles_batch.to(device)
            altitude_batch = altitude_batch.to(device)
            
            try:
                # Zero gradients for the combined optimizer
                optimizer.zero_grad()
                
                # Stage 1: Predict Chapman parameters (keep gradients)
                chapman_params = model_stage1(X_batch)
                
                # Stage 2: Generate corrections (using chapman_params with gradients)
                corrections = model_stage2(X_batch, chapman_params)
                
                # Generate Chapman profiles (keep gradients for end-to-end training)
                chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
                
                # Scale corrections
                profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
                max_correction_scale = 0.1
                corrections = corrections * profile_scale * max_correction_scale
                
                # Apply corrections to get final predictions
                predicted_profiles = chapman_profiles + corrections
                
                # Calculate losses
                stage1_loss = criterion(chapman_params, y_batch) * stage1_scale
                stage2_loss = robust_mse_loss(predicted_profiles, profiles_batch) * stage2_scale
                
                # Combined loss for end-to-end training
                total_loss = stage1_loss + stage2_loss
                
                # Check for numerical issues
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    continue
                
                # Backward pass (gradients flow through both models)
                total_loss.backward()
                
                # Check for NaN/Inf gradients
                if any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() 
                      for p in list(model_stage1.parameters()) + list(model_stage2.parameters()) 
                      if p.grad is not None):
                    continue
                
                # Clip gradients for both models
                torch.nn.utils.clip_grad_norm_(list(model_stage1.parameters()) + list(model_stage2.parameters()), max_grad_norm)
                
                # Update both models with single optimizer
                optimizer.step()
                
                # Update loss statistics
                train_loss += total_loss.item()
                stage1_loss_sum += stage1_loss.item()
                stage2_loss_sum += stage2_loss.item()
                batch_count += 1
                
            except RuntimeError as e:
                continue
        
        if batch_count > 0:
            train_loss /= batch_count
            stage1_loss_avg = stage1_loss_sum / batch_count
            stage2_loss_avg = stage2_loss_sum / batch_count
            
            train_losses.append(train_loss)
            stage1_losses.append(stage1_loss_avg)
            stage2_losses.append(stage2_loss_avg)
        
        # Validation phase
        model_stage1.eval()
        model_stage2.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch, profiles_batch, altitude_batch in val_loader:
                try:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    profiles_batch = profiles_batch.to(device)
                    altitude_batch = altitude_batch.to(device)
                    
                    # Forward pass
                    chapman_params = model_stage1(X_batch)
                    corrections = model_stage2(X_batch, chapman_params)
                    chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
                    
                    # Scale corrections
                    profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
                    max_correction_scale = 0.1
                    corrections = corrections * profile_scale * max_correction_scale
                    
                    predicted_profiles = chapman_profiles + corrections
                    
                    # Calculate losses
                    stage1_loss = criterion(chapman_params, y_batch) * stage1_scale
                    stage2_loss = robust_mse_loss(predicted_profiles, profiles_batch) * stage2_scale
                    total_loss = stage1_loss + stage2_loss
                    
                    if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                        val_loss += total_loss.item()
                        val_batch_count += 1
                        
                except RuntimeError as e:
                    continue
        
        if val_batch_count > 0:
            val_loss /= val_batch_count
            val_losses.append(val_loss)
        
        # Save best model
        if val_batch_count > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'stage1_state_dict': model_stage1.state_dict(),
                'stage2_state_dict': model_stage2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'{model_dir}/best_hybrid_model_v3.pth')
        
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Stage 1 Loss: {stage1_loss_avg:.4f}, '
                  f'Stage 2 Loss: {stage2_loss_avg:.4f}')
    
    # Save and plot training history
    save_training_history(train_losses, val_losses, stage1_losses, stage2_losses, model_dir)
    plot_training_history(train_losses, val_losses, stage1_losses, stage2_losses, model_dir)
    
    return train_losses, val_losses, stage1_losses, stage2_losses

def main():
    # Load and preprocess data
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
        # Get base features
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        f107 = f['f107'][:]
        kp = f['kp'][:]
        dip = f['dip'][:]
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]  # This is a single array of height points
        
        # Get Chapman parameters
        chapman_params = f['fit_results/chapman_params'][:]
        
        # Only use good fits
        is_good_fit = f['fit_results/is_good_fit'][:]
        latitude = latitude[is_good_fit]
        longitude = longitude[is_good_fit]
        local_time = local_time[is_good_fit]
        f107 = f107[is_good_fit]
        kp = kp[is_good_fit]
        dip = dip[is_good_fit]
        electron_density = electron_density[is_good_fit]
        chapman_params = chapman_params[is_good_fit]
        
        print(f"Number of profiles after filtering: {len(electron_density)}")
        print(f"Number of altitude points: {len(altitude)}")
    
    # Process temporal features
    temporal_features = []
    for seconds in local_time:
        dt = convert_seconds_to_datetime(seconds)
        month_sin, month_cos = circular_encode(dt['month'], 12)
        doy_sin, doy_cos = circular_encode(dt['doy'], 365)
        hour_sin, hour_cos = circular_encode(dt['hour'], 24)
        
        temporal_features.append([
            month_sin, month_cos,
            doy_sin, doy_cos,
            hour_sin, hour_cos
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
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(chapman_params)
    
    # Split the data
    indices = np.arange(len(X_scaled))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]
    y_train = y_scaled[train_idx]
    y_test = y_scaled[test_idx]
    profiles_train = electron_density[train_idx]
    profiles_test = electron_density[test_idx]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create datasets and dataloaders
    train_dataset = HybridDataset(X_train, y_train, profiles_train, altitude)
    test_dataset = HybridDataset(X_test, y_test, profiles_test, altitude)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64])
    model_stage2 = ProfileCorrector(
        profile_size=len(altitude),
        feature_size=11,
        hidden_sizes=[128, 64]
    )
    
    # Initialize single optimizer for both models (end-to-end training)
    optimizer = torch.optim.Adam(
        list(model_stage1.parameters()) + list(model_stage2.parameters()), 
        lr=0.001
    )
    
    # Train the model with end-to-end training
    train_losses, val_losses, stage1_losses, stage2_losses = train_hybrid_model_end_to_end(
        model_stage1, model_stage2,
        train_loader, test_loader,
        nn.MSELoss(),
        optimizer,
        num_epochs=30,
        device=device
    )
    
    # Evaluate and plot test results
    test_metrics = evaluate_and_plot_test_results(
        model_stage1, model_stage2,
        test_loader, device,
        './data/fit_results/hybrid_model_v3'
    )
    
    # Save the scalers
    model_dir = './data/fit_results/hybrid_model_v3'
    np.save(f'{model_dir}/X_scaler.npy', X_scaler)
    np.save(f'{model_dir}/y_scaler.npy', y_scaler)
    
    print("End-to-end training completed! Model, scalers, and test results have been saved in the hybrid_model_v3 directory.")
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4e}")

if __name__ == "__main__":
    main() 