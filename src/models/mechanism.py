"""
Pedal Mechanism Module.
Converts continuous neural network probabilities into physical damper states
using Local Context-Aware Schmitt Triggers (LCAST) and asymmetric inertia.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import maximum_filter1d

class LocalAdaptiveSchmittTrigger(nn.Module):
    """
    Local Context-Aware Schmitt Trigger (LCAST).
    Dynamically adjusts the hysteresis thresholds based on a sliding window
    of local probability peaks to prevent global masking effects.
    """
    def __init__(self, window_sec=2.0, sr=16000, hop_length=160, high_ratio=0.7, low_ratio=0.4, min_th_high=0.4):
        super().__init__()
        self.window_frames = int((window_sec * sr) / hop_length)
        if self.window_frames % 2 == 0: 
            self.window_frames += 1 
            
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.min_th_high = min_th_high

    def forward(self, x):
        batch_size, channels, time_steps = x.shape
        x_np = x.detach().cpu().numpy()
        out_np = np.zeros_like(x_np)
        
        for b in range(batch_size):
            probs = x_np[b, 0, :]
            pad_size = self.window_frames // 2
            padded_probs = np.pad(probs, (pad_size, pad_size), mode='edge')
            
            local_peaks = maximum_filter1d(padded_probs, size=self.window_frames)
            local_peaks = local_peaks[pad_size : pad_size + time_steps]
            
            current_state = 0.0
            for t in range(time_steps):
                val = probs[t]
                peak = local_peaks[t]
                
                th_high = max(self.min_th_high, peak * self.high_ratio)
                th_low = max(self.min_th_high * 0.5, peak * self.low_ratio)
                
                if val >= th_high:
                    current_state = 1.0
                elif val <= th_low:
                    current_state = 0.0
                    
                out_np[b, 0, t] = current_state
                
        return torch.from_numpy(out_np).to(x.device)

class AsymmetricInertia(nn.Module):
    """
    Simulates the physical mass and asymmetric motion delay of a piano damper.
    """
    def __init__(self, sr=16000, hop_length=160, attack_time=0.1, release_time=0.2):
        super().__init__()
        frame_rate = sr / hop_length
        self.alpha_attack = min(1.0, 1.0 / (attack_time * frame_rate))
        self.alpha_release = min(1.0, 1.0 / (release_time * frame_rate))

    def forward(self, target_curve):
        device = target_curve.device
        target_np = target_curve.detach().cpu().numpy()
        output_np = np.zeros_like(target_np)
        batch_size, _, time_steps = target_np.shape
        
        for b in range(batch_size):
            curr = 0.0
            for t in range(time_steps):
                tgt = target_np[b, 0, t]
                if tgt > curr: 
                    curr += self.alpha_attack * (tgt - curr)
                else: 
                    curr += self.alpha_release * (tgt - curr)
                output_np[b, 0, t] = curr
                
        return torch.from_numpy(output_np).to(device)

class AdvancedMechanismNoise(nn.Module):
    """
    Generates amplitude envelopes for mechanical pedal noise (press and release).
    """
    def __init__(self, noise_scale=0.1):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, pedal_curve):
        velocity = pedal_curve[..., 1:] - pedal_curve[..., :-1]
        velocity = F.pad(velocity, (1, 0))
        press_mask = F.relu(velocity)
        release_mask = F.relu(-velocity)
        
        press_gain = press_mask * self.noise_scale * 8.0
        release_gain = release_mask * self.noise_scale * 10.0
        return press_gain, release_gain

class PedalMechanism(nn.Module):
    """
    Unified mechanical module integrating LCAST, inertia, and noise envelopes.
    """
    def __init__(self, sr=16000, hop_length=160):
        super().__init__()
        self.schmitt = LocalAdaptiveSchmittTrigger()
        self.inertia = AsymmetricInertia(sr, hop_length)
        self.noise_gen = AdvancedMechanismNoise()

    def forward(self, raw_probs):
        binary = self.schmitt(raw_probs)
        position = self.inertia(binary)
        press_gain, release_gain = self.noise_gen(position)
        
        return {
            'binary': binary,
            'position': position,
            'press_gain': press_gain,
            'release_gain': release_gain
        }