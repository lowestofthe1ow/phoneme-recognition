"""
This module contains utility functions for reading input data files, like the
.wav and .txt files in datasets.
"""

import soundfile as sf


def get_wav_duration(path):
    """Returns the duration of a .wav audio file, in seconds"""
    f = sf.SoundFile(path)
    return len(f) / f.samplerate


def read_file(path):
    """Returns the contents of a file as one string"""
    with open(path, "r") as f:
        return f.read()
