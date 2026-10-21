# Third-party notices

The [MIT License](LICENSE) applies to the original project source code and associated documentation. It does not relicense third-party datasets, recordings, SoundFonts, or dependencies.

## MAESTRO v3.0.0

- **Source:** [The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), made available by Google LLC in partnership with the International Piano-e-Competition organizers.
- **License:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Attribution:** Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset*. ICLR, 2019.

Download data from the official source. This repository does not include the original dataset or a separately licensed dataset mirror. The earlier README's personal cloud-drive link has been removed in favor of the official download and local preparation scripts.

### Transformations performed by this project

- `src/prepare_dry_dataset.py` removes MIDI CC64 events and renders dry audio using FluidSynth with reverb and chorus disabled, at 16 kHz.
- `scripts/prepare_fluidsynth_wet.py` preserves MIDI CC64 and renders a baseline with reverb and chorus enabled, at 16 kHz.
- Other scripts generate NPPS outputs and excerpts of original recordings for comparison.

These files are not assigned the code's MIT license. When sharing adapted MAESTRO material, retain the source attribution and license, identify the modifications, and comply with applicable noncommercial and share-alike requirements. SoundFont terms also apply to the rendering resources. The hosted demo is a separate repository; this update does not alter its files or license notices.

## SoundFont

The local development file `piano.sf2` identifies itself in its embedded metadata as:

- **Name/version:** GeneralUser GS 1.472
- **Creator:** S. Christian Collins
- **Copyright notice:** 1997–2015 by S. Christian Collins
- **Embedded license:** License v2.0
- **Creator's website:** [schristiancollins.com](https://www.schristiancollins.com/)
- **Upstream project:** [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS)

The SoundFont binary is excluded from Git. The exact embedded license text from the local version is preserved in [`docs/licenses/GeneralUser-GS-1.472.txt`](docs/licenses/GeneralUser-GS-1.472.txt). Obtain the asset separately and retain its own notices; newer releases may have different terms or sound characteristics. The asset's metadata identifies GeneralUser GS, so it should not be described as a verified Yamaha grand-piano SoundFont.

## Software dependencies

FluidSynth and the Python packages listed in `requirements.txt` are separate projects governed by their respective upstream licenses. Installing or invoking them does not make their source code part of this project's MIT grant. Consult their distributions for the corresponding notices.

## Demo preview

`docs/assets/demo-preview.png` is a screenshot of the project's [hosted comparison interface](https://xiaofei1412.github.io/NPPS-Demo/). It contains no embedded audio; the recordings played on that site retain their applicable source licenses.
