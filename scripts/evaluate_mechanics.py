"""
Mechanistic Evaluation Script for Qualitative Quantification.
Evaluates the internal physical behaviors: Kinematic Jitter (TV, DIR) and Spectral Bloom (HFER).
"""

import sys
_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]

import os
import argparse
import yaml
import torch
import numpy as np
import librosa
from tqdm import tqdm
import torchaudio.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset import MaestroDataset
from src.models import PedalNet, PedalMechanism, AcousticModel

sys.argv = _original_argv

torch.manual_seed(42)
np.random.seed(42)

class MechanisticEvaluator:
    def __init__(self, config_path, ckpt_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.cfg = yaml.safe_load(f)
            
        print(f"[INIT] Loading Models on {self.device}...")
        self.brain = PedalNet(1, 1).to(self.device)
        self.brain.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model_state_dict'])
        self.brain.eval()
        
        self.mechanism = PedalMechanism(sr=16000, hop_length=160).to(self.device)
        self.acoustics = AcousticModel(sr=16000).to(self.device)
        self.mel_trans = T.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=229).to(self.device)

        data_root = os.path.join(PROJECT_ROOT, self.cfg['data']['root_dir'])
        dry_root = os.path.join(PROJECT_ROOT, self.cfg['data']['dry_dir'])
        self.dataset = MaestroDataset(root_dir=data_root, dry_dir=dry_root, split='test', segment_seconds=5.0)

    def calc_peak_hf_gain(self, dry_seg, wet_seg, sr=16000, cutoff_hz=2000):
        """Calculate the maximum frame-wise HF gain within the high-frequency band"""
        if len(dry_seg) < 2048:
            return 0.0
            
        # Calculate the spectrum of dry and wet sound separately
        S_dry = np.abs(librosa.stft(dry_seg, n_fft=2048, hop_length=160))**2
        S_wet = np.abs(librosa.stft(wet_seg, n_fft=2048, hop_length=160))**2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Extracting high-frequency ranges
        hf_mask = freqs > cutoff_hz
        
        # Summing along the frequency axis yields the total high-frequency energy for each frame
        hf_energy_dry_frames = np.sum(S_dry[hf_mask, :], axis=0) + 1e-9
        hf_energy_wet_frames = np.sum(S_wet[hf_mask, :], axis=0) + 1e-9
        
        # Convert to decibels (dB)
        hf_db_dry = 10 * np.log10(hf_energy_dry_frames)
        hf_db_wet = 10 * np.log10(hf_energy_wet_frames)
        
        # Calculate the gain difference for each frame and extract the peak bloom
        gain_curve = hf_db_wet - hf_db_dry
        peak_gain = np.max(gain_curve)
        
        return peak_gain

    def evaluate(self, limit_samples=None):
        print(f"[START] Running Mechanistic Evaluation on {len(self.dataset)} samples...")
        
        metrics = {
            'tv_neural': [], 'tv_phys': [],
            'dir_neural': [], 'dir_phys': [],
            'hfer_dry': [], 'hfer_npps': []
        }
        
        num_samples = limit_samples if limit_samples else len(self.dataset)
        fps = 16000 / 160
        
        with torch.no_grad():
            for idx in tqdm(range(num_samples), desc="Analyzing Internal States"):
                sample = self.dataset[idx]
                audio_dry = sample['audio'].to(self.device).unsqueeze(0)
                
                # 1. Inference Pipeline
                melspec = torch.log(self.mel_trans(audio_dry) + 1e-9)
                probs = self.brain(melspec)
                mech = self.mechanism(probs)
                
                target_len = audio_dry.shape[-1]
                pos_tensor = torch.nn.functional.interpolate(mech['position'], size=target_len, mode='linear')
                resonance = self.acoustics(audio_dry, pos_tensor)
                audio_wet = audio_dry + resonance * 1.0
                
                # Convert to numpy for metrics
                probs_np = probs.squeeze().cpu().numpy()
                pos_np = pos_tensor.squeeze().cpu().numpy()[::160] # Downsample to match frames
                audio_dry_np = audio_dry.squeeze().cpu().numpy()
                audio_wet_np = audio_wet.squeeze().cpu().numpy()
                
                # 2. Kinematic Metrics (TV & DIR)
                duration_sec = len(probs_np) / fps
                
                tv_n = np.sum(np.abs(np.diff(probs_np)))
                tv_p = np.sum(np.abs(np.diff(pos_np)))
                metrics['tv_neural'].append(tv_n)
                metrics['tv_phys'].append(tv_p)
                
                threshold_diff = 1e-4
                diff_n = np.diff(probs_np)
                diff_p = np.diff(pos_np)
                sign_n = np.sign(diff_n[np.abs(diff_n) > threshold_diff])
                sign_p = np.sign(diff_p[np.abs(diff_p) > threshold_diff])
                
                dir_n = np.sum(np.abs(np.diff(sign_n)) > 0) / duration_sec
                dir_p = np.sum(np.abs(np.diff(sign_p)) > 0) / duration_sec
                metrics['dir_neural'].append(dir_n)
                metrics['dir_phys'].append(dir_p)
                
                # 3. Acoustic Metrics (HFER) - Only during active pedal phase
                pedal_active_mask = pos_np > 0.5
                if np.sum(pedal_active_mask) > fps * 0.5: # At least 0.5s of pedal
                    # Find longest contiguous active region
                    active_indices = np.where(pedal_active_mask)[0]
                    start_frame = active_indices[0]
                    end_frame = active_indices[-1]
                    
                    start_sample = int(start_frame * 160)
                    end_sample = int(end_frame * 160)
                    
                    seg_dry = audio_dry_np[start_sample:end_sample]
                    seg_wet = audio_wet_np[start_sample:end_sample]
                    
                    peak_gain = self.calc_peak_hf_gain(seg_dry, seg_wet)
                    # Filter out abnormal negative noise
                    if peak_gain > 0:
                        metrics['hfer_npps'].append(peak_gain)

        # Report Generation
        print("\n" + "="*60)
        print("  MECHANISTIC EVALUATION REPORT (Dataset Averages)")
        print("="*60)
        
        print("[1] Kinematic Regularization (LCAST & Inertia)")
        avg_tv_n = np.mean(metrics['tv_neural'])
        avg_tv_p = np.mean(metrics['tv_phys'])
        tv_reduction = (avg_tv_n - avg_tv_p) / avg_tv_n * 100
        print(f"   - Total Variation (Neural)   : {avg_tv_n:.2f}")
        print(f"   - Total Variation (Physical) : {avg_tv_p:.2f} (Reduced by {tv_reduction:.1f}%)")
        print(f"   - Dir. Inversion Rate (Neural) : {np.mean(metrics['dir_neural']):.2f} Hz")
        print(f"   - Dir. Inversion Rate (Physical): {np.mean(metrics['dir_phys']):.2f} Hz")
        
        print("-" * 60)
        print("[2] Acoustic Reconstruction (Anti-Aliasing Exciter)")
        if metrics['hfer_npps']:
            avg_peak_gain = np.mean(metrics['hfer_npps'])
            max_peak_gain = np.max(metrics['hfer_npps'])
            print(f"   - Average Peak HF Bloom Gain (>2000Hz) : +{avg_peak_gain:.2f} dB")
            print(f"   - Maximum Observed HF Bloom Gain       : +{max_peak_gain:.2f} dB")
        else:
            print("   - Peak HF Bloom Gain: No valid data.")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ckpt', default='checkpoints/best_pedal_net.pth')
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples for quick testing")
    args = parser.parse_args()

    evaluator = MechanisticEvaluator(args.config, args.ckpt)
    evaluator.evaluate(limit_samples=args.limit)