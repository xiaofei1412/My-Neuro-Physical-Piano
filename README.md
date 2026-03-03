# Neuro-Physical Piano Sustain Synthesis 🎹

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains the official implementation of the paper:
**"Bridging the Gap: From Neural Perception to Acoustic Rendering in Piano Sustain"** *(Submitted to ISMIR)*.

## 📖 Abstract

We propose a novel Neuro-Physical Piano Synthesis (NPPS) framework that accurately models the complex acoustic phenomena of the piano sustain pedal. Unlike traditional end-to-end black-box generative models, our approach disentangles the process into a neural perception module and a deterministic physical acoustic renderer. By introducing the **Local Context-Aware Schmitt Trigger (LCAST)** and a **Dual-Stage Decay Physics Model**, we bridge the gap between abstract neural probabilities and precise mechanical damper states, achieving a Frechet Audio Distance (FAD) of 9.39 without requiring massive computational resources.

## ✨ Core Innovations

1. **PedalNet (Perception)**: A U-Net architecture estimating continuous pedal probabilities from dry Mel-spectrograms.
2. **LCAST & Asymmetric Inertia (Mechanics)**: Converts neural probabilities into realistic mechanical damper movements, preventing global masking and unnatural stuttering.
3. **Dual-Stage Decay IR (Acoustics)**: A physics-informed acoustic renderer that faithfully models the initial string punch (prompt sound) and the long-lasting soundboard resonance (after-sound), coupled with an anti-aliasing harmonic exciter.

---

## ⚙️ Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/Neuro-Physical-Pedal-Synthesis.git](https://github.com/YOUR_USERNAME/Neuro-Physical-Pedal-Synthesis.git)
cd Neuro-Physical-Pedal-Synthesis
```

**2. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**3. Install FluidSynth (Required for Data Preparation):**
FluidSynth is required to render the anechoic (dry) piano audio baseline.
* **Ubuntu/Debian:** `sudo apt-get install fluidsynth`
* **macOS:** `brew install fluidsynth`
* **Windows:** Download the binary from the [FluidSynth releases](https://github.com/FluidSynth/fluidsynth/releases) and add it to your system PATH.

*Note: You must place a valid SoundFont file named `piano.sf2` in the project root directory.*

---

## 🚀 Usage Guide

### 1. Data Preparation
We utilize the [MAESTRO v3.0.0 Dataset](https://magenta.tensorflow.org/datasets/maestro). Download and extract it into the `./data/maestro-v3.0.0` directory.
Run the surgical CC64 (pedal) extraction script to render the paired dry dataset:
```bash
python src/prepare_dry_dataset.py
```

### 2. Training the Perception Module (PedalNet)
To train the U-Net model from scratch:
```bash
python scripts/train.py --config configs/default.yaml
```
*Monitor training via TensorBoard: `tensorboard --logdir logs/`*

### 3. Inference (End-to-End Synthesis)
Synthesize a wet piano track (with physical resonance) from a single dry WAV file:
```bash
python scripts/inference.py --input path/to/dry.wav --output path/to/wet.wav --ckpt checkpoints/best_pedal_net.pth
```

### 4. Objective Evaluation
Evaluate the framework on the entire test set. Computes F1-Score, High-Frequency Ratio, RMS Energy Correlation, and FAD.
```bash
python scripts/evaluate.py --config configs/default.yaml --ckpt checkpoints/best_pedal_net.pth
```

### 5. Subjective Evaluation (Listening Test)
Generate an HTML interface containing randomized paired samples (Dry vs. Ours vs. Ground Truth) for blind MOS (Mean Opinion Score) testing:
```bash
python scripts/prepare_listening_test.py --samples 15
```
*Open `listening_test/index.html` in your browser.*

---

## 📂 Project Structure

```text
.
├── configs/
│   └── default.yaml                 # Global hyperparameters
├── src/
│   ├── models/
│   │   ├── pedal_net.py             # U-Net Perception Module
│   │   ├── mechanism.py             # LCAST & Physical Inertia
│   │   └── acoustics.py             # Dual-stage IR & Harmonic Exciter
│   ├── dataset.py                   # MAESTRO paired data loader
│   ├── losses.py                    # Optimization criteria
│   ├── utils.py                     # Logging and checkpoint utilities
│   └── prepare_dry_dataset.py       # FluidSynth rendering engine
├── scripts/
│   ├── train.py                     # Main training loop
│   ├── inference.py                 # Single-file execution
│   ├── evaluate.py                  # Objective metric benchmarking
│   ├── prepare_listening_test.py    # Subjective test HTML generator
│   └── plot_loss.py                 # Trajectory visualization
├── notebooks/
│   └── generate_paper_figures.ipynb # Scientific plotting for publication
├── requirements.txt
└── README.md
```

## 📝 Citation
If you find this code useful in your research, please consider citing our work:
```bibtex
@inproceedings{chen2026neurophysical,
  title={Bridging the Gap: From Neural Perception to Acoustic Rendering in Piano Sustain},
  author={Chen, Felix and [Co-authors]},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2026}
}
```