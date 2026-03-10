"""
FluidSynth Wet Dataset Generator (External Baseline).
Renders piano audio from MIDI files keeping all CC64 (sustain pedal) events intact.
Uses FluidSynth's default internal reverb to serve as an industry-standard baseline.
"""

import os
import pandas as pd
import subprocess
import logging
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'maestro-v3.0.0')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'data', 'maestro_wet_fluidsynth')
SF2_PATH = os.path.join(PROJECT_ROOT, 'piano.sf2')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_env():
    if not os.path.exists(SF2_PATH):
        logging.error(f"SoundFont file not found at: {SF2_PATH}")
        return False
    return True

def process_single_file(midi_rel_path):
    midi_full_path = os.path.join(DATA_ROOT, midi_rel_path)
    wav_rel_path = midi_rel_path.replace('.midi', '.wav')
    wav_full_path = os.path.join(OUTPUT_ROOT, wav_rel_path)
    
    if os.path.exists(wav_full_path):
        return
        
    os.makedirs(os.path.dirname(wav_full_path), exist_ok=True)
    
    try:
        # Standard rendering with internal reverb enabled (-R 1) and chorus enabled (-C 1)
        # We do NOT remove CC64 events here.
        cmd = [
            'fluidsynth', '-ni',
            '-g', '1.0', '-R', '1', '-C', '1', '-r', '16000',
            '-F', wav_full_path, SF2_PATH, midi_full_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.warning(f"Failed to process {midi_rel_path}: {str(e)}")

def main():
    if not check_env(): return
    logging.info("Starting FluidSynth WET Baseline Generator")
    csv_path = os.path.join(DATA_ROOT, 'maestro-v3.0.0.csv')
    df = pd.read_csv(csv_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), unit="file"):
        process_single_file(row['midi_filename'])
        
    logging.info("FluidSynth Wet baseline generation completed.")

if __name__ == "__main__":
    main()