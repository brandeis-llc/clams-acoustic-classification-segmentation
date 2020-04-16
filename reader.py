import os
import sys


def read_wavs(data_dir, file_ext=['wav'], file_per_dir=sys.maxsize):
    for r, ds, fs in os.walk(data_dir):
        for f in [f for f in fs if f.split('.')[-1] in file_ext][:file_per_dir]:
            yield r, f
