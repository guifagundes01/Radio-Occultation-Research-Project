import torch
import numpy as np

def chapmanF2F1(h, Nmax2, hmax2, H, alpha2, Nmax1, alpha1, hmax1):
    """
    Chapman function for F2 and F1 layers with improved numerical stability
    using log-space calculations to prevent overflow
    
    Parameters:
    -----------
    h : torch.Tensor
        Altitude points
    Nmax2 : torch.Tensor
        Maximum electron density for F2 layer
    hmax2 : torch.Tensor
        Height of maximum for F2 layer
    H : torch.Tensor
        Scale height
    alpha2 : torch.Tensor
        Shape parameter for F2 layer
    Nmax1 : torch.Tensor
        Maximum electron density for F1 layer
    alpha1 : torch.Tensor
        Shape parameter for F1 layer
    hmax1 : torch.Tensor
        Height of maximum for F1 layer
    """
    # # Debug input parameters
    # print("\nInput Parameters:")
    # print(f"Nmax2: {Nmax2}, hmax2: {hmax2}, H: {H}, alpha2: {alpha2}")
    # print(f"Nmax1: {Nmax1}, hmax1: {hmax1}, alpha1: {alpha1}")
    
    # Ensure all inputs are positive and within reasonable ranges
    Nmax2 = torch.clamp(Nmax2, 1e-10, 1e20)
    Nmax1 = torch.clamp(Nmax1, 1e-10, 1e20)
    H = torch.clamp(H, 1e-10, 1e20)
    alpha2 = torch.clamp(alpha2, -10, 10)
    alpha1 = torch.clamp(alpha1, -10, 10)
    
    # Calculate z terms with clipping
    z2 = torch.clamp((h - hmax2) / H, -50, 50)
    z1 = torch.clamp((h - hmax1) / H, -50, 50)
    
    # # Debug z values
    # print("\nZ values:")
    # print(f"z2 min: {z2.min()}, max: {z2.max()}")
    # print(f"z1 min: {z1.min()}, max: {z1.max()}")
    
    # Calculate exponential terms in log space
    log_ez2 = -z2
    log_ez1 = -z1
    
    
    # Calculate terms inside exponential with clipping
    term2 = torch.clamp(alpha2 * (1 - z2 - torch.exp(log_ez2)), -700, 700)
    term1 = torch.clamp(alpha1 * (1 - z1 - torch.exp(log_ez1)), -700, 700)
    
    
    # Calculate log of Nmax values
    log_Nmax2 = torch.log(Nmax2)
    log_Nmax1 = torch.log(Nmax1)
    
    
    # Calculate log of final terms
    log_term2 = log_Nmax2 + term2
    log_term1 = log_Nmax1 + term1
    
    # Use log-sum-exp trick for numerical stability
    max_log = torch.max(log_term2, log_term1)
    log_sum = max_log + torch.log(torch.exp(log_term2 - max_log) + torch.exp(log_term1 - max_log))
    
    
    # Convert back to normal space with clipping
    profile = torch.exp(torch.clamp(log_sum, -700, 700))
    
    # Final safety clip to reasonable values
    profile = torch.clamp(profile, 0, 1e8)
    
    # # Debug final profile
    # print("\nFinal profile:")
    # print(f"Profile min: {profile.min()}, max: {profile.max()}")
    # print(f"Any NaN: {torch.isnan(profile).any()}")
    # print(f"Any Inf: {torch.isinf(profile).any()}")
    
    return profile

def generate_chapman_profiles(params, altitude):
    """
    Generate Chapman profiles from parameters with improved stability
    
    Parameters:
    -----------
    params : torch.Tensor
        Chapman parameters [Nmax2, hmax2, H, alpha2, Nmax1, alpha1, hmax1]
    altitude : torch.Tensor
        Altitude points
    """
    
    # Ensure altitude is the right shape
    if len(altitude.shape) == 1:
        altitude = altitude.unsqueeze(0)
    
    # Extract and validate parameters
    Nmax2 = torch.clamp(params[:, 0], 1e-10, 1e20)
    hmax2 = params[:, 1]
    H = torch.clamp(params[:, 2], 1e-10, 1e20)
    alpha2 = torch.clamp(params[:, 3], -10, 10)
    Nmax1 = torch.clamp(params[:, 4], 1e-10, 1e20)
    alpha1 = torch.clamp(params[:, 5], -10, 10)
    hmax1 = params[:, 6]

    
    # Reshape parameters for broadcasting
    Nmax2 = Nmax2.unsqueeze(1)
    hmax2 = hmax2.unsqueeze(1)
    H = H.unsqueeze(1)
    alpha2 = alpha2.unsqueeze(1)
    Nmax1 = Nmax1.unsqueeze(1)
    alpha1 = alpha1.unsqueeze(1)
    hmax1 = hmax1.unsqueeze(1)
    
    # Generate profiles
    profiles = chapmanF2F1(altitude, Nmax2, hmax2, H, alpha2, Nmax1, alpha1, hmax1)
    
    
    return profiles 