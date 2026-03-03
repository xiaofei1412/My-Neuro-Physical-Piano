"""
Listening Test Preparation Script.
Randomly selects test segments, applies the synthesis pipeline, and generates 
an HTML interface for blind Subjective Evaluation (e.g., Mean Opinion Score).
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
import pretty_midi
import random
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.dataset import MaestroDataset
from src.models import PedalNet, PedalMechanism, AcousticModel

class ListeningTestGenerator:
    def __init__(self, config_path, ckpt_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        print(f"[Info] Initializing on {self.device}...")
        self.brain = PedalNet(1, 1).to(self.device)
        self.brain.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model_state_dict'])
        self.brain.eval()
        
        self.mechanism = PedalMechanism(sr=16000, hop_length=160).to(self.device)
        self.acoustics = AcousticModel(sr=16000).to(self.device)

    def get_active_window(self, midi_path, duration=8.0):
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            piano = pm.instruments[0]
            cc64 = [cc for cc in piano.control_changes if cc.number == 64]
            if not cc64: return 30.0 
            times = [cc.time for cc in cc64]
            start_time = times[len(times)//3]
            return start_time
        except:
            return 30.0

    def generate_test_set(self, num_samples=10, output_dir="listening_test"):
        os.makedirs(output_dir, exist_ok=True)
        data_root = self.cfg['data']['root_dir']
        dry_root = self.cfg['data'].get('dry_dir', None)
        
        import pandas as pd
        df = pd.read_csv(os.path.join(data_root, 'maestro-v3.0.0.csv'))
        test_df = df[df['split'] == 'test']
        test_samples = test_df.sample(n=num_samples, random_state=42).to_dict('records')
        
        import torchaudio.transforms as T
        mel_trans = T.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=160, n_mels=229).to(self.device)
        
        html_rows = []
        
        print("[Info] Extracting and synthesizing audio (with anti-aliasing and soft-limiting)...")
        for i, row in enumerate(tqdm(test_samples)):
            midi_rel = row['midi_filename']
            midi_path = os.path.join(data_root, midi_rel)
            real_wav_path = os.path.join(data_root, midi_rel.replace('.midi', '.wav'))
            dry_wav_path = os.path.join(dry_root, midi_rel.replace('.midi', '.wav'))
            
            start_time = self.get_active_window(midi_path, duration=8.0)
            
            try:
                audio_dry, _ = librosa.load(dry_wav_path, sr=16000, offset=start_time, duration=8.0)
                audio_real, _ = librosa.load(real_wav_path, sr=16000, offset=start_time, duration=8.0)
                
                if audio_dry.ndim > 1: audio_dry = audio_dry.mean(axis=0)
                if audio_real.ndim > 1: audio_real = audio_real.mean(axis=0)
                
                min_len = min(len(audio_dry), len(audio_real))
                audio_dry = audio_dry[:min_len]
                audio_real = audio_real[:min_len]
            except:
                continue

            audio_dry_tensor = torch.tensor(audio_dry).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                melspec = torch.log(mel_trans(audio_dry_tensor) + 1e-9)
                probs = self.brain(melspec)
                mech = self.mechanism(probs)
                
                pos = torch.nn.functional.interpolate(mech['position'], size=min_len, mode='linear')
                press = torch.nn.functional.interpolate(mech['press_gain'], size=min_len, mode='linear')
                rel = torch.nn.functional.interpolate(mech['release_gain'], size=min_len, mode='linear')
                
                resonance = self.acoustics(audio_dry_tensor, pos)
                
                raw_noise = torch.randn_like(audio_dry_tensor)
                rumble_noise = F.avg_pool1d(raw_noise, kernel_size=99, stride=1, padding=49)
                rumble_noise = rumble_noise / (torch.max(torch.abs(rumble_noise)) + 1e-9)
                mech_noise = (rumble_noise * press) + (rumble_noise * rel)
                
                wet_audio = audio_dry_tensor + resonance * 1.0 + mech_noise * 0.01
                
                # Peak Limiter
                max_val = torch.max(torch.abs(wet_audio))
                if max_val > 0.99:
                    wet_audio = wet_audio / max_val * 0.95 
                
                wet_np = wet_audio.squeeze().cpu().numpy()
                dry_np = audio_dry
                
            sample_dir = os.path.join(output_dir, f"sample_{i+1:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            sf.write(os.path.join(sample_dir, "1_dry_input.wav"), dry_np, 16000)
            sf.write(os.path.join(sample_dir, "2_ours_synthesized.wav"), wet_np, 16000)
            sf.write(os.path.join(sample_dir, "3_real_target.wav"), audio_real, 16000)
            
            html_rows.append(f"""
            <tr>
                <td>Sample {i+1}</td>
                <td><audio controls><source src="sample_{i+1:02d}/1_dry_input.wav" type="audio/wav"></audio></td>
                <td><audio controls><source src="sample_{i+1:02d}/2_ours_synthesized.wav" type="audio/wav"></audio></td>
                <td><audio controls><source src="sample_{i+1:02d}/3_real_target.wav" type="audio/wav"></audio></td>
            </tr>
            """)
            
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Neuro-Physical Piano Pedal - Listening Test</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; }}
                h1 {{ color: #333; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: white; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                th, td {{ padding: 15px; text-align: center; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                audio {{ width: 250px; }}
            </style>
        </head>
        <body>
            <h1>Neuro-Physical Piano Pedal Synthesis</h1>
            <p>Subjective Evaluation Interface: Compare the synthesized acoustic resonance against the dry input and real piano recording.</p>
            <table>
                <tr>
                    <th>Sample ID</th>
                    <th>1. Dry Input (FluidSynth)</th>
                    <th>2. Ours (Synthesized Resonance)</th>
                    <th>3. Ground Truth (Real Piano)</th>
                </tr>
                {"".join(html_rows)}
            </table>
        </body>
        </html>
        """
        
        html_path = os.path.join(output_dir, "index.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"\n[SUCCESS] Listening test index generated at: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ckpt', default='checkpoints/best_pedal_net.pth')
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    ListeningTestGenerator(os.path.join(PROJECT_ROOT, args.config), os.path.join(PROJECT_ROOT, args.ckpt)).generate_test_set(num_samples=args.samples)