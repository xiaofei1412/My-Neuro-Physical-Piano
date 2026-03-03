"""
MAESTRO Dataset Loader.
Handles the alignment and loading of paired dry acoustic signals (synthesized) 
and ground-truth pedal control curves extracted from MIDI files.
"""

import os
import torch
import pandas as pd
import soundfile as sf
import pretty_midi
import numpy as np
import logging
from torch.utils.data import Dataset

class MaestroDataset(Dataset):
    """
    Dataset class for Neuro-Physical Piano Synthesis.
    Forces mono audio loading to handle stereo synthesizer outputs uniformly.
    """
    def __init__(self, root_dir, dry_dir=None, split='train', segment_seconds=4.0, sr=16000):
        self.root_dir = root_dir
        self.dry_dir = dry_dir
        self.split = split
        self.segment_seconds = segment_seconds
        self.sr = sr
        self.segment_length = int(sr * segment_seconds)
        
        csv_path = os.path.join(root_dir, 'maestro-v3.0.0.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
            
        self.meta_data = pd.read_csv(csv_path)
        
        if split:
            self.meta_data = self.meta_data[self.meta_data['split'] == split]
            
        logging.info(f"Initialized {split} set with {len(self.meta_data)} files.")
        if self.dry_dir:
            logging.info(f"Using Paired Data: Input from {self.dry_dir}")

    def __len__(self):
        return len(self.meta_data)

    def _load_pedal_curve(self, midi_path, duration_sec):
        """Extracts continuous CC64 (sustain pedal) values from MIDI."""
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
            target_len = int(duration_sec * self.sr)
            pedal_curve = np.zeros(target_len, dtype=np.float32)
            
            piano = pm.instruments[0]
            cc64_events = [cc for cc in piano.control_changes if cc.number == 64]
            cc64_events.sort(key=lambda x: x.time)
            
            current_val = 0.0
            last_idx = 0
            
            for cc in cc64_events:
                time_sec = cc.time
                val_norm = cc.value / 127.0
                idx = int(time_sec * self.sr)
                
                if idx >= target_len:
                    pedal_curve[last_idx:] = current_val
                    break
                
                pedal_curve[last_idx:idx] = current_val
                current_val = val_norm
                last_idx = idx
            
            if last_idx < target_len:
                pedal_curve[last_idx:] = current_val
                
            return pedal_curve
        except Exception as e:
            logging.warning(f"Error reading MIDI {midi_path}: {e}")
            return np.zeros(int(duration_sec * self.sr), dtype=np.float32)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        midi_filename = row['midi_filename']
        midi_path = os.path.join(self.root_dir, midi_filename)
        
        if self.dry_dir:
            wav_filename = midi_filename.replace('.midi', '.wav')
            audio_path = os.path.join(self.dry_dir, wav_filename)
        else:
            audio_path = os.path.join(self.root_dir, midi_filename.replace('.midi', '.wav'))

        if not os.path.exists(audio_path):
            return {
                'audio': torch.zeros(1, self.segment_length), 
                'pedal': torch.zeros(1, self.segment_length)
            }

        try:
            info = sf.info(audio_path)
            file_len = info.frames
            
            if file_len > self.segment_length:
                max_start = file_len - self.segment_length
                start_frame = np.random.randint(0, max_start)
                
                audio, _ = sf.read(audio_path, start=start_frame, stop=start_frame+self.segment_length, dtype='float32')
                
                if audio.ndim == 2:
                    audio = audio.mean(axis=1) 
                
                pedal_full = self._load_pedal_curve(midi_path, info.duration)
                if len(pedal_full) < start_frame + self.segment_length:
                    pad_len = (start_frame + self.segment_length) - len(pedal_full)
                    pedal_full = np.pad(pedal_full, (0, pad_len))
                pedal = pedal_full[start_frame : start_frame + self.segment_length]
                
            else:
                audio, _ = sf.read(audio_path, dtype='float32')
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                
                pedal = self._load_pedal_curve(midi_path, info.duration)
                
                pad_len = self.segment_length - len(audio)
                audio = np.pad(audio, (0, pad_len))
                pedal = np.pad(pedal, (0, pad_len))

            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            pedal_tensor = torch.from_numpy(pedal).float().unsqueeze(0)
            
            return {
                'audio': audio_tensor, 
                'pedal': pedal_tensor,
                'name': midi_filename
            }
            
        except Exception as e:
            logging.error(f"Processing error for {midi_filename}: {e}")
            return {
                'audio': torch.zeros(1, self.segment_length), 
                'pedal': torch.zeros(1, self.segment_length)
            }