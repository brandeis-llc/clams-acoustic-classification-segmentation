import os

import librosa
import numpy as np

MFCC_SIZE = 40

labels = {'speech': 0, 'music': 1, 'noise': 2}


def extract(wav_fname, normalize=False, **kwargs):
    print(f'extracting: {wav_fname}\t', end='')
    audio, sr = librosa.load(wav_fname, **kwargs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_SIZE).T
    print(librosa.get_duration(audio, sr), mfccs.shape)
    if normalize:
        return cmvn(mfccs)
    else:
        return mfccs


def cmvn(mfccs):
    raise NotImplementedError


def index_label(label_str, binary=True):
    if label_str in labels:
        label_idx = labels[label_str]
        if binary:
            label_idx = min(1, label_idx)
        return label_idx
    else:
        return -1


def extract_all(wav_paths, train=False, binary_class=True):
    features, labels = np.empty((0, MFCC_SIZE)), np.empty(0)
    label = None
    for wav_dir, wav_fname in wav_paths:
        if train:
            label_str = wav_fname.split('-')[0]
            label = index_label(label_str, binary_class)
        full_fname = os.path.join(wav_dir, wav_fname)
        feature = extract(full_fname)
        labels = np.append(labels, [label] * len(feature))
        features = np.vstack([features, feature])
    return np.array(features), np.array(labels, dtype=np.int)
