import os

import librosa
import numpy as np

MFCC_SIZE = 40


def fn_to_mfcc(wav_fname, normalize=False, **kwargs):
    audio, sr = librosa.load(wav_fname, **kwargs)
    print(librosa.get_duration(audio, sr))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_SIZE).T
    print(mfccs.shape)
    if normalize:
        return cmvn(mfccs)
    else:
        return mfccs


def cmvn(mfccs):
    raise NotImplementedError


def read_wavs(data_dir, file_ext='.wav'):
    features, labels = np.empty((0, MFCC_SIZE)), np.empty(0)
    for r, ds, fs in os.walk(data_dir):
        if len(ds) == 0:
            # means leaf dirs
            label_str = r.split(os.sep)[-1]
            label = np.zeros if label_str == "speech" else np.ones
            for f in [f for f in fs if f.endswith(file_ext)]:
                abs_fname = os.path.join(r, f)
                feature = fn_to_mfcc(abs_fname)
                labels = np.append(labels, label(len(feature)))
                features = np.vstack([features, feature])
    return np.array(features), np.array(labels, dtype=np.int)


if __name__ == '__main__':
    import sys
    X, Y = read_wavs(sys.argv[1])

