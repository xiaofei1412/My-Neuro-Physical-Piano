# Research record and reproducibility notes

NPPS originated as an undergraduate final-year project, *Neuro-Physical Piano Synthesis: A Full-Pipeline System for Sustain Pedal Detection and Acoustic Rendering*. Its accompanying unpublished manuscript is titled *Mitigating Physical Incoherence in Data-Driven Piano Synthesis via Mechano-Acoustic Constraints* (Xiaofei Chen and Shengchen Li, 2026).

This page preserves the distinction between historical manuscript measurements and the behavior of the code currently distributed. The presentation update does not change model behavior or rerun training, listening tests, or dataset evaluation.

## Historical manuscript measurements

The following numbers were reported in the manuscript. They have not been revalidated against the current code and data during this repository update and should not be treated as a verified benchmark or a state-of-the-art claim.

| Manuscript method label | TV ↓ | DIR (Hz) ↓ | Peak HF gain | RMS correlation | FAD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| FluidSynth wet baseline | — | — | — | 0.2274 | 11.0834 |
| Without LCAST | 12.80 | 74.62 | +10.85 dB | 0.0143 | 9.8843 |
| Without dual-stage decay | 0.90 | 0.00 | +0.65 dB | 0.0062 | 10.0246 |
| Full NPPS | 0.90 | 0.00 | +11.12 dB | 0.0136 | 9.8853 |

The FluidSynth wet baseline consumes the original MIDI pedal timings. The public demo instead compares **dry FluidSynth input, NPPS output, and a real piano recording**. Its third column is not the FluidSynth wet baseline.

## Current implementation

| Component | Behavior in the distributed source |
| --- | --- |
| Supervision | Normalized continuous MIDI CC64 values; `MSE + 0.1 × L1` in `scripts/train.py`. |
| Dataset split | The `split` column of the original MAESTRO metadata is used. |
| Training defaults | 16 kHz audio, 4-second crops, batch size 16, 100 epochs, AdamW, cosine learning-rate schedule. |
| Mechanical processing | A centered local-peak window of approximately 2 seconds, followed by hysteresis; default inertia attack/release constants are 0.1/0.2 seconds. |
| Gradient flow | LCAST and inertia detach tensors and run NumPy processing on the CPU. They do not backpropagate into PedalNet. |
| Acoustic processing | Fixed frequency-dependent impulse-response decay and harmonic-exciter parameters. |

The manuscript's BCE-based training description, some hyperparameters, and its discussion of tensorized differentiable mechanics do not match this source snapshot. The configured resume-checkpoint field is not consumed by the current training loop. Inference and evaluation construct their own feature transforms, so matching all preprocessing settings to a training run requires checking the relevant scripts.

## Evaluation caveats

### Audio alignment and sample rates

`MaestroDataset.__getitem__` randomly crops dry audio, including for the test split. `scripts/evaluate.py` reads the reference recording and FluidSynth baseline from sample zero instead of the same crop offset. It also ignores their original sample rates when reading and writes the reference at 16 kHz without resampling. Original MAESTRO recordings may be at 44.1 or 48 kHz.

Consequently, historical RMS correlation and FAD comparisons need to be rerun with matched time intervals and correctly resampled references. RMS correlation is computed against the real recording in the current evaluator. FAD is a distribution-level embedding metric; it does not independently establish accurate pedal timing or physical validity. The evaluator also silently skips failed items and reuses output folders; a validated run must account for the processed sample set and avoid stale files.

### Meaning of the ablations

- `--ablation no_lcast` replaces LCAST with a global threshold, while **retaining asymmetric inertia**.
- `--ablation no_dual_stage` substitutes single-stage decay in the impulse response, while **retaining the harmonic exciter**.
- `scripts/evaluate_mechanics.py` compares raw continuous predictions with the post-mechanism trajectory; this is not identical to the `no_lcast` pipeline.

These modes should not be labeled as removing the entire mechanical stage or the entire acoustic stage. The historical manuscript table is retained above without claiming that every cell is reproducible from the present flags.

### Mechanistic indicators and illustrations

TV and DIR quantify the behavior of a control trajectory; smoothing alone is not evidence of correct pedal detection. The implemented DIR calculation counts thresholded derivative sign changes per second and does not implement the manuscript's stated 50 ms window criterion. Peak high-frequency gain measures added energy relative to dry audio, not reconstruction accuracy. The current script restricts analysis to a span containing active-pedal frames and only aggregates positive gains.

The notebook includes **synthetically perturbed controls** and **manually controlled step responses**. Its acoustic illustration also uses a different resonance mixing gain from the main inference script. These figures demonstrate component responses and should not be interpreted as direct visualizations of trained-model predictions on the test set. A low-pass filter after nonlinear excitation also does not, by itself, establish that all aliasing has been eliminated.

### Listening study and demo

The manuscript reports a preliminary study with 12 listeners. `scripts/plot_mos.py` contains manually entered score arrays; participant-level collection records and analysis provenance are not provided in this repository. One manuscript passage reports an artifact score of 4.33 while the corresponding plotted bar is 4.42. These measurements have not been independently reconstructed during this update.

The generated listening interface and the hosted demo show labeled audio columns. They support exploratory listening and should not be described as a randomized double-blind experiment.

## Assets and reproducibility status

- Dataset files, SoundFonts, generated audio, and model checkpoints are excluded from Git.
- No public checkpoint download is supplied in this repository; training or a separately supplied compatible checkpoint is required for inference.
- Dataset preparation additionally requires the FluidSynth executable and a separately licensed SoundFont. The local SoundFont metadata identifies GeneralUser GS 1.472, rather than a verified Yamaha-specific bank.
- Dependencies use lower-bound version constraints rather than a frozen environment. A historical training environment and its exact package versions are not supplied here.
- Centered context and the offline processing path preclude a claim of strictly causal real-time inference.

The current source supports inspection and further experimentation. Exact reproduction of historical claims requires resolving the configuration and evaluation issues above, retaining data/model provenance, and rerunning the experiments. See [third-party notices](../THIRD_PARTY_NOTICES.md) for asset licensing.
