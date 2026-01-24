"""
This module contains utility functions for reading input data files, like the
.wav and .txt files in datasets.
"""

import wave
import contextlib


def get_wav_duration(path):
    """Returns the duration of a .wav audio file, in seconds"""
    with contextlib.closing(wave.open(str(path), "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def read_file(path):
    """Returns the contents of a file as one string"""
    with open(path, "r") as f:
        return f.read()
