import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_training_history():
    """Plot the training and validation loss history"""
    try:
        # Try to load from numpy files first
        train_losses = np.load('./data/fit_results/train_losses.npy')
        val_losses = np.load('./data/fit_results/val_losses.npy')
    except:
        # If numpy files don't exist, try to load from checkpoint
        checkpoint = torch.load('./data/fit_results/best_chapman_mlp.pth')
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig('./data/fit_results/training_history.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_training_history() 