"""
Loss functions for probabilistic forecasting with masking support.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import math

try:
    import scoringrules as sr
    HAS_SCORINGRULES = True
except ImportError:
    HAS_SCORINGRULES = False
    print("Warning: scoringrules package not available. MaskedCRPSEnsemble will not work.")


class MaskedCRPSNormal(nn.Module):
    """
    Continuous Ranked Probability Score for Normal distribution.
    Masks out NaN values in targets before computing loss.

    Args:
        pred: Distribution object with .loc (mean) and .scale (std) attributes
        y: Target values, may contain NaN
    """

    def __init__(self):
        super(MaskedCRPSNormal, self).__init__()

    def forward(self, pred, y):
        mask = ~torch.isnan(y)
        y = y[mask]
        mu = pred.loc[mask].flatten()
        sigma = pred.scale[mask].flatten()

        normal = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(sigma))

        scaled = (y - mu) / sigma

        Phi = normal.cdf(scaled)
        phi = torch.exp(normal.log_prob(scaled))

        crps = sigma * (scaled * (2 * Phi - 1) + 2 * phi - (1 /
                        torch.sqrt(torch.tensor(torch.pi, device=sigma.device))))

        return crps.mean()


class MaskedCRPSLogNormal(nn.Module):
    """
    Continuous Ranked Probability Score for LogNormal distribution.
    Useful for wind speed and other strictly positive quantities.

    Based on: Baran and Lerch (2015) "Log-normal distribution based Ensemble 
    Model Output Statistics models for probabilistic wind-speed forecasting"

    Args:
        pred: Distribution object with .loc (mu) and .scale (sigma) attributes
        y: Target values, may contain NaN
        t: Optional time index to compute loss for single timestep
        L: Total number of lead times (default: 97)
    """

    def __init__(self):
        super(MaskedCRPSLogNormal, self).__init__()
        self.i = 0
        self.eps = 1e-5

    def _prepare_inputs(self, pred, y, t=None):
        if y.dim() == 2:  # [B, N] 
            mask = ~torch.isnan(y)
            y_masked = y[mask]
            
            mu = pred.loc  # [B, N]
            sigma = pred.scale 
            
            mu = mu[mask].flatten()
            sigma = sigma[mask].flatten()
        elif y.dim() == 4:  # [B, L, N, 1] - all horizons
            B, L, N, _ = y.shape
            mask = ~torch.isnan(y)
            y_masked = y[mask]
            
            mu = pred.loc
            sigma = pred.scale

            if mu.dim() == 3:  # [B, L, N]
                mu = mu.unsqueeze(-1)  # [B, L, N, 1]
                sigma = sigma.unsqueeze(-1)  # [B, L, N, 1]

            if t is not None:
                mu = mu[:, t, :, :].unsqueeze(1)
                sigma = sigma[:, t, :, :].unsqueeze(1)

            mu = mu[mask].flatten()
            sigma = sigma[mask].flatten()
        else:
            raise ValueError(f"Unexpected y shape: {y.shape}. Expected [B, N] or [B, L, N, 1]")
            
        return mu, sigma, y_masked

    def forward(self, pred, y, t=None, L=97):
        mu, sigma, y_masked = self._prepare_inputs(pred, y, t)

        y_masked = y_masked + self.eps

        normal = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(sigma))

        omega = (torch.log(y_masked) - mu) / sigma
        ex_input = mu + (sigma**2) / 2

        ex_input = torch.clamp(ex_input, max=15)
        self.i += 1

        ex = 2 * torch.exp(ex_input)

        crps = y_masked * (2 * normal.cdf(omega) - 1.0) - \
            ex * (normal.cdf(omega - sigma) + normal.cdf(sigma / (2**0.5)) - 1.0)

        return crps.mean()



class MaskedMAE(nn.Module):
    """
    Masked Mean Absolute Error.
    Only computes error on finite (non-NaN, non-Inf) target values.

    Returns:
        loss: Mean absolute error
        valid: Number of valid elements used in computation
    """

    def __init__(self):
        super(MaskedMAE, self).__init__()

    def forward(self, pred, target):
        mask = torch.isfinite(target)
        valid = mask.sum()

        if valid == 0:
            return torch.tensor(float('nan'), device=pred.device), valid

        abs_err = (pred[mask] - target[mask]).abs()
        return abs_err.mean(), valid


class MaskedTwCRPSLogNormal(MaskedCRPSLogNormal):
    """
    Threshold-Weighted CRPS for LogNormal distribution (Right Tail).
    Weights the CRPS by w(z) = 1{z >= threshold}.
    Computation: CRPS_total - Integral_0_to_threshold(CRPS_integrand).
    
    Args:
        threshold: The threshold value. Regions z >= threshold are weighted 1, z < threshold weighted 0.
    """
    def __init__(self, threshold=10.0):
        super().__init__()
        self.threshold = float(threshold)
        self.num_integration_steps = 100

    def forward(self, pred, y, t=None):
        # 1. Compute Standard CRPS (Full Domain)
        # Using parent logic
        mu, sigma, y_masked = self._prepare_inputs(pred, y, t)
        y_masked = y_masked + self.eps

        normal = torch.distributions.Normal(
            torch.zeros_like(mu), torch.ones_like(sigma))

        omega = (torch.log(y_masked) - mu) / sigma
        ex_input = mu + (sigma**2) / 2
        ex_input = torch.clamp(ex_input, max=15)
        
        ex = 2 * torch.exp(ex_input)

        crps_full = y_masked * (2 * normal.cdf(omega) - 1.0) - \
            ex * (normal.cdf(omega - sigma) + normal.cdf(sigma / (2**0.5)) - 1.0)

        # 2. Compute Left Tail Integral [0, threshold]
        # twCRPS(z>=thr) = CRPS_total - CRPS(z<thr)
        
        if self.threshold <= 0:
            return crps_full.mean()
            
        # Integration grid z: [Steps, 1]
        z = torch.linspace(1e-4, self.threshold, self.num_integration_steps, device=y.device).unsqueeze(1) # [S, 1]
        dz = z[1,0] - z[0,0]
        
        # Broadcast params: [1, M]
        mu_b = mu.unsqueeze(0)
        sigma_b = sigma.unsqueeze(0)
        y_b = y_masked.unsqueeze(0)
        
        # LogNormal CDF at z: F(z) = Phi((ln(z) - mu)/sigma)
        # Note: z is positive (starts at 1e-4) to avoid log(0)
        standardized_z = (torch.log(z) - mu_b) / sigma_b
        F_z = normal.cdf(standardized_z) # [S, M]
        
        # Heaviside: H(z - y) = 1 if z >= y else 0
        H_z_y = (z >= y_b).float()
        
        integrand = (F_z - H_z_y).pow(2)
        
        # Trapezoidal rule over z
        # sum ( (I[:-1] + I[1:]) / 2 * dz )
        integral = (integrand[:-1] + integrand[1:]).sum(dim=0) * (dz / 2.0) # [M]
        
        tw_crps = crps_full - integral
        
        return tw_crps.mean()