import torch
import math

def calibrate_gaussian_noise(epsilon, delta, clip_norm):
    """
    Calibrate Gaussian noise scale given privacy budget (epsilon, delta).

    Args:
        epsilon (float): privacy budget ε (use float("inf") for no DP).
        delta (float): target δ (e.g., 1e-5).
        clip_norm (float): clipping bound (L2 sensitivity).

    Returns:
        float: stddev of Gaussian noise to add.
    """
    if math.isinf(epsilon):
        return 0.0
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("epsilon must be > 0 (or inf), delta in (0,1).")
    # σ ≥ sqrt(2 log(1.25/δ)) * Δ2 / ε
    sigma = math.sqrt(2 * math.log(1.25 / delta)) * clip_norm / epsilon
    return sigma


def add_dp_noise(latents, epsilon=float("inf"), delta=1e-5, clip_norm=1.0, device="cpu"):
    """
    Add Gaussian DP noise to latent vectors.

    Args:
        latents (Tensor): shape (N, d), latent vectors.
        epsilon (float): privacy budget ε (default inf = no DP).
        delta (float): δ parameter of (ε,δ)-DP.
        clip_norm (float): clipping bound for L2 norm (sensitivity).
        device (str): device for noise tensor.

    Returns:
        Tensor: noisy, clipped latents.
    """
    # --- Clip each latent vector to L2 norm <= clip_norm ---
    latents_flat = latents.view(latents.size(0), -1)
    norms = latents_flat.norm(2, dim=1, keepdim=True)
    scaling = clip_norm / (norms + 1e-6)
    scaling = torch.minimum(torch.ones_like(scaling), scaling)
    latents_clipped = latents_flat * scaling
    latents_clipped = latents_clipped.view_as(latents)

    # --- Compute noise scale ---
    sigma = calibrate_gaussian_noise(epsilon, delta, clip_norm)

    if sigma > 0:
        noise = torch.normal(
            mean=0.0, std=sigma, size=latents.shape, device=device
        )
        return latents_clipped + noise
    else:
        return latents_clipped
