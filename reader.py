import os


def read_wavs(data_dir, file_ext='.wav'):
    for r, ds, fs in os.walk(data_dir):
        if len(ds) == 0:
            for f in [f for f in fs if f.endswith(file_ext)]:
                yield r, f


