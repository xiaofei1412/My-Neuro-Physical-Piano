"""
Single-file inference script.
Applies the neuro-physical synthesis pipeline to a user-provided dry audio file.
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
import torchaudio.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models import PedalNet, PedalMechanism, AcousticModel

class PianoSynthesizer:
    """End-to-end synthesizer wrapping perception, mechanics, and acoustics."""
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.cfg = yaml.safe_load(f)
        
        self.brain = PedalNet(1, 1).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.brain.load_state_dict(ckpt['model_state_dict'])
        self.brain.eval()
        
        self.mechanism = PedalMechanism(sr=self.cfg['data']['sr'], hop_length=self.cfg['data']['hop_length']).to(self.device)
        self.acoustics = AcousticModel(sr=self.cfg['data']['sr']).to(self.device)

    def infer(self, input_path, output_path):
        audio, sr = librosa.load(input_path, sr=self.cfg['data']['sr'])
        if audio.ndim == 2: 
            audio = audio.mean(axis=1)
        audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)
        
        mel_trans = T.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=160, n_mels=229).to(self.device)
        melspec = torch.log(mel_trans(audio_tensor) + 1e-9)
        
        with torch.no_grad():
            probs = self.brain(melspec)
            mech = self.mechanism(probs)
            
            target_len = audio_tensor.shape[-1]
            pos = torch.nn.functional.interpolate(mech['position'], size=target_len, mode='linear')
            press = torch.nn.functional.interpolate(mech['press_gain'], size=target_len, mode='linear')
            rel = torch.nn.functional.interpolate(mech['release_gain'], size=target_len, mode='linear')
            
            resonance = self.acoustics(audio_tensor, pos)
            
            # Generate low-frequency rumble
            raw_noise = torch.randn_like(audio_tensor)
            rumble_noise = F.avg_pool1d(raw_noise, kernel_size=99, stride=1, padding=49)
            rumble_noise = rumble_noise / (torch.max(torch.abs(rumble_noise)) + 1e-9)
            mech_noise = (rumble_noise * press) + (rumble_noise * rel)
            
            # Final mixing with peak limiting
            wet_audio = audio_tensor + resonance * 1.0 + mech_noise * 0.01
            
            max_val = torch.max(torch.abs(wet_audio))
            if max_val > 0.99: 
                wet_audio = wet_audio / max_val * 0.95
                
        sf.write(output_path, wet_audio.squeeze().cpu().numpy(), sr)
        print(f"Synthesized audio saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neuro-Physical Piano Synthesis")
    parser.add_argument('--input', type=str, required=True, help="Path to input dry WAV file")
    parser.add_argument('--output', type=str, default='output_wet.wav', help="Path to output WAV file")
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_pedal_net.pth', help="Model checkpoint")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Config file")
    args = parser.parse_args()
    
    synthesizer = PianoSynthesizer(os.path.join(PROJECT_ROOT, args.ckpt), os.path.join(PROJECT_ROOT, args.config))
    synthesizer.infer(args.input, args.output)