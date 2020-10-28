import os
import sys


def read_wavs(data_dir, file_ext=['wav'], file_per_dir=sys.maxsize):
    for r, ds, fs in os.walk(data_dir):
        for f in fs[:file_per_dir]:
            if f.split('.')[-1] in file_ext:
                yield r, f
