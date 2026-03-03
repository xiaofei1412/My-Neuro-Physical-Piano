"""
Objective Evaluation Script.
Computes comprehensive metrics including Pedal Control F1-Score, 
High-Frequency Ratio, RMS Correlation, and Frechet Audio Distance (FAD).
Outputs a standardized comparative benchmark report.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from frechet_audio_distance import FrechetAudioDistance

# Global seed lock to ensure exact reproducibility in acoustic physics modules
torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset import MaestroDataset
from src.models import PedalNet, PedalMechanism, AcousticModel

class Evaluator:
    def __init__(self, config_path, ckpt_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.cfg = yaml.safe_load(f)
        
        print(f"[EVAL] Initializing Evaluation on {self.device}...")
        
        self.brain = PedalNet(in_channels=1, out_channels=1).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.brain.load_state_dict(ckpt['model_state_dict'])
        self.brain.eval()
        
        self.mechanism = PedalMechanism(sr=self.cfg['data']['sr'], hop_length=self.cfg['data']['hop_length']).to(self.device)
        self.acoustics = AcousticModel(sr=self.cfg['data']['sr']).to(self.device)
        self.frechet = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False, verbose=False)

    def safe_rms_corr(self, a, b):
        ra = librosa.feature.rms(y=a, hop_length=512).flatten()
        rb = librosa.feature.rms(y=b, hop_length=512).flatten()
        m = min(len(ra), len(rb))
        if np.std(ra[:m]) < 1e-6 or np.std(rb[:m]) < 1e-6: return 0.0
        return pearsonr(ra[:m], rb[:m])[0]

    def safe_hf_ratio(self, a):
        s = np.abs(librosa.stft(a))
        f = librosa.fft_frequencies(sr=16000)
        total = np.sum(s)
        if total < 1e-6: return 0.0
        return np.sum(s[f > 2500]) / total

    def run_evaluation(self, output_dir="eval_results"):
        ref_dir = os.path.join(output_dir, "ref_real") 
        dry_dir_out = os.path.join(output_dir, "baseline_dry")
        est_dir = os.path.join(output_dir, "est_wet")
        
        for d in [ref_dir, dry_dir_out, est_dir]: os.makedirs(d, exist_ok=True)
        
        dataset = MaestroDataset(root_dir=self.cfg['data']['root_dir'], dry_dir=self.cfg['data'].get('dry_dir'), split='test', segment_seconds=5.0)
        
        # Separated tracking for dry and wet metrics to allow unified comparative reporting
        metrics = {
            'f1': [], 
            'hf_ratio_dry': [], 
            'hf_ratio_wet': [], 
            'rms_corr_dry': [], 
            'rms_corr_wet': []
        }
        
        print(f"[EVAL] Processing FULL Test Set: {len(dataset)} samples...")
        import torchaudio.transforms as T
        mel_trans = T.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=229).to(self.device)
        
        for idx in tqdm(range(len(dataset))):
            try:
                sample = dataset[idx]
                audio_dry = sample['audio'].to(self.device).unsqueeze(0)
                pedal_gt = sample['pedal'].numpy().flatten()
                name = f"{idx:04d}.wav"
                
                with torch.no_grad():
                    melspec = torch.log(mel_trans(audio_dry) + 1e-9)
                    probs = self.brain(melspec)
                    mech = self.mechanism(probs)
                    
                    target_len = audio_dry.shape[-1]
                    pos = torch.nn.functional.interpolate(mech['position'], size=target_len, mode='linear')
                    press = torch.nn.functional.interpolate(mech['press_gain'], size=target_len, mode='linear')
                    rel = torch.nn.functional.interpolate(mech['release_gain'], size=target_len, mode='linear')
                    
                    resonance = self.acoustics(audio_dry, pos)
                    
                    raw_noise = torch.randn_like(audio_dry)
                    rumble_noise = F.avg_pool1d(raw_noise, kernel_size=99, stride=1, padding=49)
                    rumble_noise = rumble_noise / (torch.max(torch.abs(rumble_noise)) + 1e-9)
                    mech_noise = (rumble_noise * press) + (rumble_noise * rel)
                    
                    wet_audio = audio_dry + resonance * 1.0 + mech_noise * 0.01
                    max_val = torch.max(torch.abs(wet_audio))
                    if max_val > 0.99:
                        wet_audio = wet_audio / max_val * 0.95
                    
                    wet_np = np.nan_to_num(wet_audio.squeeze().cpu().numpy())
                    dry_np = np.nan_to_num(audio_dry.squeeze().cpu().numpy())

                # Control Evaluation
                pred_bin = (probs.squeeze().cpu().numpy() > 0.5).astype(int)
                from scipy.ndimage import zoom
                gt_res = zoom(pedal_gt, len(pred_bin)/len(pedal_gt), order=0) > 0.5 if len(pred_bin) != len(pedal_gt) else pedal_gt > 0.5
                metrics['f1'].append(f1_score(gt_res, pred_bin, zero_division=0))
                
                # Spectral Bloom Evaluation
                metrics['hf_ratio_dry'].append(self.safe_hf_ratio(dry_np))
                metrics['hf_ratio_wet'].append(self.safe_hf_ratio(wet_np))
                
                # Energy Correlation Evaluation
                real_path = os.path.join(dataset.root_dir, dataset.meta_data.iloc[idx]['midi_filename'].replace('.midi', '.wav'))
                if os.path.exists(real_path):
                    real_audio, _ = sf.read(real_path, start=0, stop=len(wet_np))
                    if real_audio.ndim == 2: real_audio = real_audio.mean(axis=1)
                    metrics['rms_corr_dry'].append(self.safe_rms_corr(real_audio, dry_np))
                    metrics['rms_corr_wet'].append(self.safe_rms_corr(real_audio, wet_np))
                    sf.write(os.path.join(ref_dir, name), real_audio, 16000)
                
                sf.write(os.path.join(est_dir, name), wet_np, 16000)
                sf.write(os.path.join(dry_dir_out, name), dry_np, 16000)

            except Exception as e: 
                continue

        # --- Standardized Benchmark Report Formatting ---
        print("\n" + "="*60)
        print("          FULL DATASET EVALUATION REPORT          ")
        print("="*60)
        
        print("1. [CONTROL] Pedal Prediction F1-Score:")
        print(f"   - Model Accuracy: {np.mean(metrics['f1']):.4f}")
        print("-" * 60)
        
        avg_hf_dry = np.mean(metrics['hf_ratio_dry'])
        avg_hf_wet = np.mean(metrics['hf_ratio_wet'])
        print("2. [PHYSICS] High-Frequency Ratio (>2500Hz):")
        print(f"   - Baseline (Dry): {avg_hf_dry:.4f} | Ours (Wet): {avg_hf_wet:.4f}")
        print(f"   - Absolute Boost: {(avg_hf_wet - avg_hf_dry):+.4f}")
        print("-" * 60)
        
        avg_rms_dry = np.mean(metrics['rms_corr_dry'])
        avg_rms_wet = np.mean(metrics['rms_corr_wet'])
        print("3. [PHYSICS] RMS Energy Correlation (vs. Real Piano):")
        print(f"   - Baseline (Dry): {avg_rms_dry:.4f} | Ours (Wet): {avg_rms_wet:.4f}")
        print(f"   - Improvement:    {(avg_rms_wet - avg_rms_dry):+.4f}")
        print("-" * 60)
        
        print("4. [AUDIO]   Frechet Audio Distance (FAD):")
        try:
            fad_dry = self.frechet.score(ref_dir, dry_dir_out)
            fad_wet = self.frechet.score(ref_dir, est_dir)
            print(f"   - Baseline (Dry): {fad_dry:.4f} | Ours (Wet): {fad_wet:.4f}")
            print(f"   - Improvement:    {(fad_dry - fad_wet):+.4f}") # Positive is better
        except Exception as e: 
            print(f"   - Evaluation failed: {e}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Neuro-Physical Synthesis metrics.")
    parser.add_argument('--config', default='configs/default.yaml', help="Path to config file")
    parser.add_argument('--ckpt', default='checkpoints/best_pedal_net.pth', help="Model checkpoint")
    args = parser.parse_args()
    Evaluator(os.path.join(PROJECT_ROOT, args.config), os.path.join(PROJECT_ROOT, args.ckpt)).run_evaluation()