"""
Acoustic Rendering Module.
Simulates the acoustic properties of a grand piano soundboard and strings 
using a dual-stage decay impulse response and anti-aliasing harmonic exciter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal

class HarmonicExciter(nn.Module):
    """
    Anti-aliasing non-linear harmonic exciter.
    Applies soft saturation to mid-frequencies while filtering out 
    high-frequency digital artifacts (aliasing).
    """
    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr
        self.register_buffer('bp_kernel', self._make_bandpass_kernel())
        self.register_buffer('lp_kernel', self._make_lowpass_kernel())

    def _make_bandpass_kernel(self):
        N = 31
        nyq = self.sr / 2
        b = scipy.signal.firwin(N, [400/nyq, 4000/nyq], pass_zero=False)
        return torch.from_numpy(b).float().view(1, 1, -1)
        
    def _make_lowpass_kernel(self):
        N = 31
        b = scipy.signal.firwin(N, 7000/(self.sr/2), pass_zero=True)
        return torch.from_numpy(b).float().view(1, 1, -1)

    def forward(self, x, intensity):
        mids = F.conv1d(x, self.bp_kernel, padding=15)
        saturated = F.softsign(mids * 2.0)
        harmonics = F.conv1d(saturated, self.lp_kernel, padding=15)
        return harmonics * intensity * 0.05

class PinkNoiseIR(nn.Module):
    """
    Generates a physics-informed impulse response utilizing dual-stage decay
    envelopes (prompt sound and after-sound) and specific body resonances.
    """
    def __init__(self, sr=16000, duration=4.0):
        super().__init__()
        self.sr = sr
        self.duration = duration
        ir = self._generate_ir()
        self.register_buffer('ir', torch.from_numpy(ir).float().view(1, 1, -1))

    def _generate_pink_noise(self, n_samples):
        np.random.seed(42) 
        uneven = n_samples % 2
        X = np.random.randn(n_samples // 2 + 1 + uneven) + 1j * np.random.randn(n_samples // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 0.8) 
        noise = (np.fft.irfft(X / S)).real
        if uneven: 
            noise = noise[:-1]
        return noise / (np.max(np.abs(noise)) + 1e-9)

    def _generate_ir(self):
        n_samples = int(self.sr * self.duration)
        noise = self._generate_pink_noise(n_samples)
        nyquist = self.sr / 2
        
        # High-pass filter to prevent low-frequency build-up (muddy sound)
        b_hp, a_hp = scipy.signal.butter(2, 100 / nyquist, btype='high')
        noise = scipy.signal.lfilter(b_hp, a_hp, noise)
        
        t = np.linspace(0, self.duration, n_samples)
        
        # Frequency band splitting
        b_low, a_low = scipy.signal.butter(2, 200/nyquist, btype='low')
        b_mid, a_mid = scipy.signal.butter(2, [200/nyquist, 3000/nyquist], btype='band')
        b_high, a_high = scipy.signal.butter(2, 3000/nyquist, btype='high')
        
        noise_low = scipy.signal.lfilter(b_low, a_low, noise)
        noise_mid = scipy.signal.lfilter(b_mid, a_mid, noise)
        noise_high = scipy.signal.lfilter(b_high, a_high, noise)
        
        # Dual-stage decay modeling
        decay_low = 0.7 * np.exp(-6.0 * t) + 0.3 * np.exp(-1.5 * t)
        decay_mid = 0.5 * np.exp(-5.0 * t) + 0.5 * np.exp(-2.5 * t)
        decay_high = 0.8 * np.exp(-8.0 * t) + 0.2 * np.exp(-4.0 * t)
        
        ir = (noise_low * decay_low * 0.35) + \
             (noise_mid * decay_mid * 0.75) + \
             (noise_high * decay_high * 0.25)
             
        def apply_peak(sig, freq, q, gain):
            b, a = scipy.signal.iirpeak(freq/nyquist, q, gain)
            return scipy.signal.lfilter(b, a, sig)
            
        # Presence and warmth resonance peaks
        ir = apply_peak(ir, 400, 1.5, 1.5)  
        ir = apply_peak(ir, 1200, 2.0, 2.0) 
        
        return ir / (np.max(np.abs(ir)) + 1e-9)

    def forward(self):
        return self.ir

class AcousticModel(nn.Module):
    """
    End-to-end acoustic processor combining body resonance via FFT convolution
    and spectral bloom via harmonic excitation.
    """
    def __init__(self, sr=16000):
        super().__init__()
        self.pink_ir = PinkNoiseIR(sr=sr)
        self.exciter = HarmonicExciter(sr=sr)
        
    def fft_convolve(self, signal, kernel):
        n_step = signal.shape[-1]
        n_kernel = kernel.shape[-1]
        n_fft = 2 ** int(np.ceil(np.log2(n_step + n_kernel - 1)))
        signal_f = torch.fft.rfft(signal, n=n_fft)
        kernel_f = torch.fft.rfft(kernel, n=n_fft)
        return torch.fft.irfft(signal_f * kernel_f, n=n_fft)[..., :n_step]

    def forward(self, dry_audio, pedal_control):
        ir = self.pink_ir()
        body = self.fft_convolve(dry_audio, ir)
        
        # Safe gain staging to preserve headroom
        body = body * 0.045 
        
        bloom = self.exciter(dry_audio, pedal_control)
        total_resonance = (body * pedal_control) + bloom
        return total_resonance