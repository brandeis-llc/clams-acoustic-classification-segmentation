import os

import librosa
import numpy as np

MFCC_SIZE = 40

labels = {'speech': 0, 'music': 1, 'noise': 2}


def extract(wav_fname, normalize=False, verbose=True, **kwargs):
    if verbose:
        print(f'extracting: {wav_fname}\t', end='', flush=True)
    # will sample 16000 points per second
    audio, sr = librosa.load(wav_fname, sr=16000, **kwargs)
    # then slide over frames of 1/100 seconds (=10ms)
    frame_sliding_size = sr // 100
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=MFCC_SIZE, hop_length=frame_sliding_size).T
    if verbose:
        print(librosa.get_duration(audio, sr), mfccs.shape, flush=True)
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


def extract_all(wav_paths, train=False, binary_class=True, persist=False):
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
    data = (np.array(features), np.array(labels, dtype=np.int))
    if persist:
        import datetime
        timestamp = datetime.datetime.today().strftime('%Y%m%d-%H%M')
        np.savez(f'_models/{timestamp}.features.{features.shape[1]}d', xs=data[0], ys=data[1])
    return data
