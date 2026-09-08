# DrumTranscriber (not included)

This folder is intentionally empty in the repository.

The optional `drumtranscriber` onset mode of the pipeline uses
**DrumTranscriber** by yoshi-man, a CNN-based drum-hit classifier
(Hi-hat, Crash, Kick, Snare, Ride, Toms):

> https://github.com/yoshi-man/DrumTranscriber

The upstream repository does **not** declare a licence, so its code cannot be
redistributed here. To use the drumtranscriber onset mode, download it
yourself:

```bash
git clone https://github.com/yoshi-man/DrumTranscriber.git
cp -r DrumTranscriber/* drumtranscriber/
```

so that `drumtranscriber/DrumTranscriber.py` and `drumtranscriber/model/`
exist, and install its dependencies:

```bash
pip install librosa tensorflow numpy pandas scikit-learn
```

Note: the thesis results were produced **without** this component (onset
mode `librosa`), so it is not required to reproduce them.
