import os
from pathlib import Path

# Get absolute path to drumtranscriber directory
_DRUMTRANSCRIBER_DIR = Path(__file__).parent.parent.absolute()

SETTINGS = {
    "LABELS_INDEX": {
        0: 'crash',
        1: 'hihat_c',
        2: 'kick_drum',
        3: 'ride',
        4: 'snare',
        5: 'tom_h'
    },
    'TARGET_SHAPE': (256, 256),
    'SAVED_MODEL_PATH': str(_DRUMTRANSCRIBER_DIR / "model" / "drum_transcriber.h5")
}
