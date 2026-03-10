"""
Objective Evaluation Script with Ablation Support.
Evaluates NPPS against external baselines and internal ablated versions.
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

torch.manual_seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset import MaestroDataset
from src.models import PedalNet, PedalMechanism, AcousticModel

class Evaluator:
    def __init__(self, config_path, ckpt_path, ablation, fluidsynth_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.cfg = yaml.safe_load(f)
            
        self.ablation = ablation
        self.fluidsynth_dir = fluidsynth_dir
        
        # Parse Ablation Flags
        use_lcast = False if ablation == 'no_lcast' else True
        use_dual_stage = False if ablation == 'no_dual_stage' else True
        
        print(f"[EVAL] Mode: {ablation.upper()} | LCAST: {use_lcast} | Dual-Stage: {use_dual_stage}")
        
        self.brain = PedalNet(1, 1).to(self.device)
        self.brain.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model_state_dict'])
        self.brain.eval()
        
        self.mechanism = PedalMechanism(sr=16000, hop_length=160, use_lcast=use_lcast).to(self.device)
        self.acoustics = AcousticModel(sr=16000, use_dual_stage=use_dual_stage).to(self.device)
        self.frechet = FrechetAudioDistance(model_name="vggish", sample_rate=16000, use_pca=False, use_activation=False, verbose=False)

    def safe_rms_corr(self, a, b):
        ra = librosa.feature.rms(y=a, hop_length=512).flatten()
        rb = librosa.feature.rms(y=b, hop_length=512).flatten()
        m = min(len(ra), len(rb))
        if np.std(ra[:m]) < 1e-6 or np.std(rb[:m]) < 1e-6: return 0.0
        return pearsonr(ra[:m], rb[:m])[0]

    def run_evaluation(self, output_dir="eval_results"):
        ref_dir = os.path.join(output_dir, "ref_real") 
        est_dir = os.path.join(output_dir, f"est_{self.ablation}")
        os.makedirs(ref_dir, exist_ok=True)
        os.makedirs(est_dir, exist_ok=True)
        
        dataset = MaestroDataset(root_dir=self.cfg['data']['root_dir'], dry_dir=self.cfg['data'].get('dry_dir'), split='test', segment_seconds=5.0)
        
        metrics = {'rms_ours': []}
        if self.fluidsynth_dir: metrics['rms_fluid'] = []
        
        print(f"[EVAL] Processing Test Set: {len(dataset)} samples...")
        import torchaudio.transforms as T
        mel_trans = T.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=229).to(self.device)
        
        for idx in tqdm(range(len(dataset))):
            try:
                sample = dataset[idx]
                audio_dry = sample['audio'].to(self.device).unsqueeze(0)
                name = f"{idx:04d}.wav"
                
                # Render Ours
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
                    if max_val > 0.99: wet_audio = wet_audio / max_val * 0.95
                    wet_np = np.nan_to_num(wet_audio.squeeze().cpu().numpy())

                # Load Ground Truth
                real_path = os.path.join(dataset.root_dir, dataset.meta_data.iloc[idx]['midi_filename'].replace('.midi', '.wav'))
                if os.path.exists(real_path):
                    real_audio, _ = sf.read(real_path, start=0, stop=len(wet_np))
                    if real_audio.ndim == 2: real_audio = real_audio.mean(axis=1)
                    sf.write(os.path.join(ref_dir, name), real_audio, 16000)
                    metrics['rms_ours'].append(self.safe_rms_corr(real_audio, wet_np))
                
                # Load FluidSynth Baseline
                if self.fluidsynth_dir:
                    fluid_path = os.path.join(self.fluidsynth_dir, dataset.meta_data.iloc[idx]['midi_filename'].replace('.midi', '.wav'))
                    if os.path.exists(fluid_path):
                        fluid_audio, _ = sf.read(fluid_path, start=0, stop=len(wet_np))
                        if fluid_audio.ndim == 2: fluid_audio = fluid_audio.mean(axis=1)
                        metrics['rms_fluid'].append(self.safe_rms_corr(real_audio, fluid_audio))
                
                sf.write(os.path.join(est_dir, name), wet_np, 16000)
                
            except Exception as e: 
                continue

        # --- Report ---
        print("\n" + "="*60)
        print(f"   EVALUATION REPORT: {self.ablation.upper()}   ")
        print("="*60)
        print(f"[PHYSICS] RMS Energy Correlation (vs. Real Piano):")
        print(f"   - Ours ({self.ablation}): {np.mean(metrics['rms_ours']):.4f}")
        if self.fluidsynth_dir and metrics['rms_fluid']:
            print(f"   - External (FluidSynth Wet): {np.mean(metrics['rms_fluid']):.4f}")
            
        print("-" * 60)
        print("[AUDIO] Frechet Audio Distance (FAD):")
        try:
            fad_ours = self.frechet.score(ref_dir, est_dir)
            print(f"   - Ours ({self.ablation}): {fad_ours:.4f}")
            if self.fluidsynth_dir:
                fad_fluid = self.frechet.score(ref_dir, self.fluidsynth_dir) # Compare real with all fluidsynth
                print(f"   - External (FluidSynth Wet): {fad_fluid:.4f}")
        except Exception as e: 
            print(f"   - FAD Computation Failed: {e}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ckpt', default='checkpoints/best_pedal_net.pth')
    parser.add_argument('--ablation', type=str, choices=['full', 'no_lcast', 'no_dual_stage'], default='full', help="Ablation mode")
    parser.add_argument('--fluidsynth_dir', type=str, default='', help="Path to maestro_wet_fluidsynth dir for external comparison")
    args = parser.parse_args()
    
    fs_dir = os.path.join(PROJECT_ROOT, args.fluidsynth_dir) if args.fluidsynth_dir else None
    Evaluator(os.path.join(PROJECT_ROOT, args.config), os.path.join(PROJECT_ROOT, args.ckpt), args.ablation, fs_dir).run_evaluation()