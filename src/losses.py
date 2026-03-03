"""
Loss functions module.
Provides robust optimization criteria for both control curve regression 
and multi-scale spectral audio distance.
"""

import torch
import torch.nn as nn

class PedalLoss(nn.Module):
    """
    Hybrid loss function for Pedal Curve Regression.
    Combines Mean Squared Error (MSE) and L1 Loss for robust optimization.
    """
    def __init__(self, mse_weight=1.0, l1_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0.0
        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse(pred, target)
        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1(pred, target)
        return loss

class MultiScaleSpectralLoss(nn.Module):
    """
    Computes the spectral distance between generated and target audio 
    across multiple FFT resolutions. Essential for differentiable DSP optimization.
    """
    def __init__(self, fft_sizes=[2048, 1024, 512, 256, 128, 64], loss_type='L1'):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type

    def forward(self, x_pred, x_true):
        loss = 0.0
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window_pred = torch.hann_window(n_fft).to(x_pred.device)
            window_true = torch.hann_window(n_fft).to(x_true.device)
            
            S_pred = torch.stft(x_pred, n_fft, hop, return_complex=True, window=window_pred)
            S_true = torch.stft(x_true, n_fft, hop, return_complex=True, window=window_true)
            
            S_pred_mag = torch.abs(S_pred)
            S_true_mag = torch.abs(S_true)
            
            if self.loss_type == 'L1':
                loss += (S_pred_mag - S_true_mag).abs().mean()
            else:
                loss += ((S_pred_mag - S_true_mag)**2).mean()
                
            loss += (torch.log(S_pred_mag + 1e-7) - torch.log(S_true_mag + 1e-7)).abs().mean()
            
        return loss