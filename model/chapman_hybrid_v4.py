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
    local_time_date = pd.to_datetime(seconds, unit='s')
    return {
        'year': local_time_date.year,
        'month': local_time_date.month,
        'doy': local_time_date.dayofyear,
        'hour': local_time_date.hour,
        'minute': local_time_date.minute
    }

def circular_encode(value, max_value):
    sin_val = np.sin(2 * np.pi * value / max_value)
    cos_val = np.cos(2 * np.pi * value / max_value)
    return sin_val, cos_val

class ChapmanPredictor(nn.Module):
    def __init__(self, input_size=11, hidden_sizes=[128, 64], output_size=7):
        super(ChapmanPredictor, self).__init__()
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
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class ProfileCorrector(nn.Module):
    def __init__(self, profile_size, feature_size, hidden_sizes=[128, 64]):
        super(ProfileCorrector, self).__init__()
        combined_size = feature_size + 7
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
        layers.append(nn.Linear(prev_size, profile_size))
        layers.append(nn.Tanh())
        self.dense = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, features, chapman_params):
        combined = torch.cat([features, chapman_params], dim=1)
        corrections = self.dense(combined)
        return corrections

class HybridDataset(Dataset):
    def __init__(self, X, y, profiles, altitude):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.profiles = torch.FloatTensor(profiles)
        self.altitude = torch.FloatTensor(altitude)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.profiles[idx], self.altitude

def save_training_history(train_losses, val_losses, stage1_losses, stage2_losses, model_dir):
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'stage1_losses': stage1_losses,
        'stage2_losses': stage2_losses
    }
    np.save(f'{model_dir}/train_losses.npy', train_losses)
    np.save(f'{model_dir}/val_losses.npy', val_losses)
    np.save(f'{model_dir}/stage1_losses.npy', stage1_losses)
    np.save(f'{model_dir}/stage2_losses.npy', stage2_losses)
    with open(f'{model_dir}/training_history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=4)

def plot_training_history(train_losses, val_losses, stage1_losses, stage2_losses, model_dir):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(stage1_losses, label='Stage 1')
    plt.title('Stage 1 Loss (Chapman Parameters)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(stage2_losses, label='Stage 2')
    plt.title('Stage 2 Loss (Profile Corrections)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
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
    model_stage1.eval()
    model_stage2.eval()
    all_true_profiles = []
    all_pred_profiles = []
    all_altitudes = []
    with torch.no_grad():
        for X_batch, y_batch, profiles_batch, altitude_batch in test_loader:
            X_batch = X_batch.to(device)
            profiles_batch = profiles_batch.to(device)
            altitude_batch = altitude_batch.to(device)
            chapman_params = model_stage1(X_batch)
            chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
            corrections = model_stage2(X_batch, chapman_params)
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            predicted_profiles = chapman_profiles + corrections
            all_true_profiles.append(profiles_batch.cpu().numpy())
            all_pred_profiles.append(predicted_profiles.cpu().numpy())
            all_altitudes.append(altitude_batch.cpu().numpy())
    all_true_profiles = np.concatenate(all_true_profiles, axis=0)
    all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
    all_altitudes = np.concatenate(all_altitudes, axis=0)
    mse = np.mean((all_true_profiles - all_pred_profiles) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_true_profiles - all_pred_profiles))
    rel_error = np.abs(all_true_profiles - all_pred_profiles) / (np.abs(all_true_profiles) + 1e-8)
    mean_rel_error = np.mean(rel_error)
    median_rel_error = np.median(rel_error)
    plt.figure(figsize=(15, 10))
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
    plt.subplot(2, 2, 2)
    plt.hist(rel_error.flatten(), bins=50, alpha=0.7)
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    plt.subplot(2, 2, 3)
    mean_true = np.mean(all_true_profiles, axis=0)
    mean_pred = np.mean(all_pred_profiles, axis=0)
    std_true = np.std(all_true_profiles, axis=0)
    std_pred = np.std(all_pred_profiles, axis=0)
    plt.semilogy(all_altitudes[0], mean_true, 'b-', label='True Mean')
    plt.semilogy(all_altitudes[0], mean_pred, 'r--', label='Predicted Mean')
    plt.fill_between(all_altitudes[0], mean_true - std_true, mean_true + std_true, alpha=0.2, color='blue')
    plt.fill_between(all_altitudes[0], mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, color='red')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Electron Density (cm⁻³)')
    plt.title('Mean Profile Comparison')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.loglog(all_true_profiles.flatten(), all_pred_profiles.flatten(), 'k.', alpha=0.1)
    plt.loglog([1e9, 1e13], [1e9, 1e13], 'r--')
    plt.xlabel('True Electron Density (cm⁻³)')
    plt.ylabel('Predicted Electron Density (cm⁻³)')
    plt.title('True vs Predicted')
    plt.grid(True)
    metrics_text = f'MSE: {mse:.2e}\nRMSE: {rmse:.2e}\nMAE: {mae:.2e}\nMean Rel Error: {mean_rel_error:.2%}\nMedian Rel Error: {median_rel_error:.2%}'
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f'{model_dir}/test_results.png')
    plt.close()
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

def train_phase1_stage1_only(model_stage1, train_loader, val_loader, criterion, optimizer, num_epochs=30, device='cuda'):
    """Train ChapmanPredictor (Stage 1) only."""
    model_stage1 = model_stage1.to(device)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model_stage1.train()
        train_loss, batch_count = 0, 0
        for X_batch, y_batch, _, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model_stage1(X_batch)
            loss = criterion(pred, y_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            train_losses.append(train_loss / batch_count)
        # Validation
        model_stage1.eval()
        val_loss, val_batch_count = 0, 0
        with torch.no_grad():
            for X_batch, y_batch, _, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model_stage1(X_batch)
                loss = criterion(pred, y_batch)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batch_count += 1
        if val_batch_count > 0:
            val_losses.append(val_loss / val_batch_count)
        if val_batch_count > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_stage1.state_dict(), './data/fit_results/hybrid_model_v4/stage1_best.pth')
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    return train_losses, val_losses

def train_phase2_stage2_only(model_stage1, model_stage2, train_loader, val_loader, criterion, optimizer, num_epochs=30, device='cuda'):
    """Freeze Stage 1 and train ProfileCorrector (Stage 2) only."""
    model_stage1 = model_stage1.to(device)
    model_stage2 = model_stage2.to(device)
    for param in model_stage1.parameters():
        param.requires_grad = False
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model_stage2.train()
        train_loss, batch_count = 0, 0
        for X_batch, y_batch, profiles_batch, altitude_batch in train_loader:
            X_batch = X_batch.to(device)
            profiles_batch = profiles_batch.to(device)
            altitude_batch = altitude_batch.to(device)
            with torch.no_grad():
                chapman_params = model_stage1(X_batch)
                chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
            optimizer.zero_grad()
            corrections = model_stage2(X_batch, chapman_params)
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            corrected_profiles = chapman_profiles + corrections
            rel_error = torch.abs(corrected_profiles - profiles_batch) / (torch.abs(profiles_batch) + 1e-8)
            rel_error = torch.clamp(rel_error, 0, 10.0)
            mse = torch.mean(rel_error ** 2)
            loss = mse
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            train_losses.append(train_loss / batch_count)
        # Validation
        model_stage2.eval()
        val_loss, val_batch_count = 0, 0
        with torch.no_grad():
            for X_batch, y_batch, profiles_batch, altitude_batch in val_loader:
                X_batch = X_batch.to(device)
                profiles_batch = profiles_batch.to(device)
                altitude_batch = altitude_batch.to(device)
                chapman_params = model_stage1(X_batch)
                chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
                corrections = model_stage2(X_batch, chapman_params)
                profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
                max_correction_scale = 0.1
                corrections = corrections * profile_scale * max_correction_scale
                corrected_profiles = chapman_profiles + corrections
                rel_error = torch.abs(corrected_profiles - profiles_batch) / (torch.abs(profiles_batch) + 1e-8)
                rel_error = torch.clamp(rel_error, 0, 10.0)
                mse = torch.mean(rel_error ** 2)
                loss = mse
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batch_count += 1
        if val_batch_count > 0:
            val_losses.append(val_loss / val_batch_count)
        if val_batch_count > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_stage2.state_dict(), './data/fit_results/hybrid_model_v4/stage2_best.pth')
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    for param in model_stage1.parameters():
        param.requires_grad = True
    return train_losses, val_losses

def train_phase3_finetune(model_stage1, model_stage2, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """Unfreeze both stages and fine-tune end-to-end with low learning rate."""
    model_stage1 = model_stage1.to(device)
    model_stage2 = model_stage2.to(device)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model_stage1.train()
        model_stage2.train()
        train_loss, batch_count = 0, 0
        for X_batch, y_batch, profiles_batch, altitude_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            profiles_batch = profiles_batch.to(device)
            altitude_batch = altitude_batch.to(device)
            optimizer.zero_grad()
            chapman_params = model_stage1(X_batch)
            corrections = model_stage2(X_batch, chapman_params)
            chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
            profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
            max_correction_scale = 0.1
            corrections = corrections * profile_scale * max_correction_scale
            predicted_profiles = chapman_profiles + corrections
            stage1_loss = criterion(chapman_params, y_batch) * 0.3
            rel_error = torch.abs(predicted_profiles - profiles_batch) / (torch.abs(profiles_batch) + 1e-8)
            rel_error = torch.clamp(rel_error, 0, 10.0)
            mse = torch.mean(rel_error ** 2)
            stage2_loss = mse * 1.0
            total_loss = stage1_loss + stage2_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            batch_count += 1
        if batch_count > 0:
            train_losses.append(train_loss / batch_count)
        # Validation
        model_stage1.eval()
        model_stage2.eval()
        val_loss, val_batch_count = 0, 0
        with torch.no_grad():
            for X_batch, y_batch, profiles_batch, altitude_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                profiles_batch = profiles_batch.to(device)
                altitude_batch = altitude_batch.to(device)
                chapman_params = model_stage1(X_batch)
                corrections = model_stage2(X_batch, chapman_params)
                chapman_profiles = generate_chapman_profiles(chapman_params, altitude_batch)
                profile_scale = torch.mean(torch.abs(chapman_profiles), dim=1, keepdim=True)
                max_correction_scale = 0.1
                corrections = corrections * profile_scale * max_correction_scale
                predicted_profiles = chapman_profiles + corrections
                stage1_loss = criterion(chapman_params, y_batch) * 0.3
                rel_error = torch.abs(predicted_profiles - profiles_batch) / (torch.abs(profiles_batch) + 1e-8)
                rel_error = torch.clamp(rel_error, 0, 10.0)
                mse = torch.mean(rel_error ** 2)
                stage2_loss = mse * 1.0
                total_loss = stage1_loss + stage2_loss
                if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                    val_loss += total_loss.item()
                    val_batch_count += 1
        if val_batch_count > 0:
            val_losses.append(val_loss / val_batch_count)
        if val_batch_count > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'stage1_state_dict': model_stage1.state_dict(),
                'stage2_state_dict': model_stage2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './data/fit_results/hybrid_model_v4/finetune_best.pth')
        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    return train_losses, val_losses

def main():
    # Load and preprocess data (same as v2/v3)
    with h5py.File('./data/filtered/electron_density_profiles_2023_with_fits.h5', 'r') as f:
        latitude = f['latitude'][:]
        longitude = f['longitude'][:]
        local_time = f['local_time'][:]
        f107 = f['f107'][:]
        kp = f['kp'][:]
        dip = f['dip'][:]
        electron_density = f['electron_density'][:]
        altitude = f['altitude'][:]
        chapman_params = f['fit_results/chapman_params'][:]
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
    X = np.column_stack([
        latitude,
        longitude,
        temporal_features,
        f107,
        kp,
        dip
    ])
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(chapman_params)
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
    train_dataset = HybridDataset(X_train, y_train, profiles_train, altitude)
    test_dataset = HybridDataset(X_test, y_test, profiles_test, altitude)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_stage1 = ChapmanPredictor(input_size=11, hidden_sizes=[128, 64])
    model_stage2 = ProfileCorrector(profile_size=len(altitude), feature_size=11, hidden_sizes=[128, 64])
    # Phase 1: Train ChapmanPredictor only
    optimizer1 = torch.optim.Adam(model_stage1.parameters(), lr=0.001)
    print("\n--- Phase 1: Training ChapmanPredictor (Stage 1) only ---")
    train_phase1_stage1_only(model_stage1, train_loader, test_loader, nn.MSELoss(), optimizer1, num_epochs=30, device=device)
    # Phase 2: Freeze Stage 1, train ProfileCorrector only
    optimizer2 = torch.optim.Adam(model_stage2.parameters(), lr=0.001)
    print("\n--- Phase 2: Training ProfileCorrector (Stage 2) only ---")
    train_phase2_stage2_only(model_stage1, model_stage2, train_loader, test_loader, nn.MSELoss(), optimizer2, num_epochs=30, device=device)
    # Phase 3: Fine-tune both stages end-to-end with low learning rate
    optimizer_finetune = torch.optim.Adam(list(model_stage1.parameters()) + list(model_stage2.parameters()), lr=1e-4)
    print("\n--- Phase 3: Fine-tuning both stages end-to-end ---")
    train_phase3_finetune(model_stage1, model_stage2, train_loader, test_loader, nn.MSELoss(), optimizer_finetune, num_epochs=10, device=device)
    # Evaluate and plot test results
    test_metrics = evaluate_and_plot_test_results(model_stage1, model_stage2, test_loader, device, './data/fit_results/hybrid_model_v4')
    # Save the scalers
    model_dir = './data/fit_results/hybrid_model_v4'
    np.save(f'{model_dir}/X_scaler.npy', X_scaler)
    np.save(f'{model_dir}/y_scaler.npy', y_scaler)
    print("\nThree-phase training completed! Model, scalers, and test results have been saved in the hybrid_model_v4 directory.")
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4e}")

if __name__ == "__main__":
    main() 