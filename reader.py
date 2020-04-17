import os
import sys


def read_wavs(data_dir, file_ext=['wav'], file_per_dir=sys.maxsize):
    for f in [f for f in os.listdir(data_dir) if f.split('.')[-1] in file_ext][:file_per_dir]:
        yield data_dir, f
