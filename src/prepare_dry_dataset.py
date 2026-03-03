"""
Dry Dataset Generator.
Utilizes FluidSynth to render anechoic (dry) piano audio segments from MIDI files, 
surgically removing all CC64 (sustain pedal) events to establish a clean baseline.
"""

import os
import pandas as pd
import pretty_midi
import subprocess
import logging
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'maestro-v3.0.0')
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'data', 'maestro_dry')
SF2_PATH = os.path.join(PROJECT_ROOT, 'piano.sf2')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_env():
    """Validates the local environment dependencies."""
    if not os.path.exists(SF2_PATH):
        logging.error(f"SoundFont file not found at: {SF2_PATH}")
        return False
    try:
        subprocess.run(['fluidsynth', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        logging.error("FluidSynth executable not found in system PATH.")
        return False
    return True

def process_single_file(midi_rel_path):
    """Processes a single MIDI file into a dry acoustic waveform."""
    midi_full_path = os.path.join(DATA_ROOT, midi_rel_path)
    wav_rel_path = midi_rel_path.replace('.midi', '.wav')
    wav_full_path = os.path.join(OUTPUT_ROOT, wav_rel_path)
    
    if os.path.exists(wav_full_path):
        return
        
    os.makedirs(os.path.dirname(wav_full_path), exist_ok=True)
    
    try:
        pm = pretty_midi.PrettyMIDI(midi_full_path)
        
        # Surgical extraction of pedal events
        for inst in pm.instruments:
            inst.control_changes = [cc for cc in inst.control_changes if cc.number != 64]
            
        temp_midi_path = wav_full_path.replace('.wav', '_temp_no_pedal.mid')
        pm.write(temp_midi_path)
        
        # Anechoic rendering configuration
        cmd = [
            'fluidsynth', '-ni',
            '-g', '1.0', '-R', '0', '-C', '0', '-r', '16000',
            '-F', wav_full_path, SF2_PATH, temp_midi_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(temp_midi_path):
            os.remove(temp_midi_path)
            
    except Exception as e:
        logging.warning(f"Failed to process {midi_rel_path}: {str(e)}")

def main():
    if not check_env():
        return

    logging.info("Starting Dry Dataset Generator (FluidSynth Backend)")
    csv_path = os.path.join(DATA_ROOT, 'maestro-v3.0.0.csv')
    
    if not os.path.exists(csv_path):
        logging.error(f"Metadata index not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded metadata containing {len(df)} entries. Initiating rendering...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), unit="file"):
        process_single_file(row['midi_filename'])
        
    logging.info("Dry dataset generation completed successfully.")

if __name__ == "__main__":
    main()