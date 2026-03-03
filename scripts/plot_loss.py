"""
Training Loss Visualization Tool.
Parses experiment logs to plot training and validation trajectories.
"""

import os
import matplotlib.pyplot as plt
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_log_and_plot(log_dir_name):
    log_folder = os.path.join(PROJECT_ROOT, "logs", log_dir_name)
    if not os.path.exists(log_folder):
        print(f"[Error] Log directory not found: {log_folder}")
        return

    log_files = [f for f in os.listdir(log_folder) if f.endswith('.log')]
    if not log_files:
        print("[Error] No .log files found in the directory.")
        return
    
    log_files.sort(reverse=True)
    latest_log = os.path.join(log_folder, log_files[0])
    print(f"[Info] Reading log file: {latest_log}")

    train_losses = []
    val_losses = []
    epochs = []

    pattern = re.compile(r"Epoch (\d+) \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)")

    with open(latest_log, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))

    if not epochs:
        print("[Warning] No epoch-level loss data found in the log.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    
    plt.title(f"Training Trajectory - {log_dir_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE + L1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    exp_name = 'pedal_unet_v1' 
    parse_log_and_plot(exp_name)