import os

import librosa
import numpy as np

labels = {'speech': 0, 'music': 1, 'noise': 2}


def extract(wav_fname, frame_context=0, zcr=False, mfccbin=20, verbose=True, **kwargs):
    # will sample 16000 points per second
    audio, sr = librosa.load(wav_fname, sr=16000, **kwargs)
    if verbose:
        print(f'feature extracting: {wav_fname}\t', end='', flush=True)
    feats = spectral_feats(audio, sr, mfccbin, zcr)
    if frame_context > 0:
        feats = temporal_feat(feats, frame_context)
    if verbose:
        print(f'length: {librosa.get_duration(audio, sr)} secs, frames: {feats.shape}', end='', flush=True)
        print('', flush=True)
    return feats


def spectral_feats(audio, samplerate, mfccbin, zcr):
    # then slide over frames of 1/100 seconds (=10ms)
    frame_sliding_size = samplerate // 100
    feats = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=mfccbin, hop_length=frame_sliding_size)
    if zcr:
        zcrs = librosa.feature.zero_crossing_rate(y=audio, hop_length=frame_sliding_size)
        feats = np.concatenate((feats, zcrs), axis=0)
    # transpose so that rows are time frames
    return feats.T


def temporal_feat(frames, frame_context):
    last_frame = len(frames)
    temporalized_frames = None
    for i in range(last_frame):
        # +1 to the end as array slicing is exclusive
        context = frames[max(0, i-frame_context):min(last_frame, i+frame_context)+1]
        means = np.mean(context, axis=0)
        vars = np.var(context, axis=0)
        stds = np.std(context, axis=0)
        temporalized_frame = np.concatenate((means, vars, stds), axis=0)
        if temporalized_frames is None:
            temporalized_frames = np.empty((0, len(temporalized_frame)))
        temporalized_frames = np.vstack([temporalized_frames, temporalized_frame])
    return temporalized_frames


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


def extract_all(wav_paths, frame_context=0, train=False, binary_class=True, persist=False):
    features = None
    labels = np.empty(0)
    label = None
    for wav_dir, wav_fname in wav_paths:
        if train:
            label_str = wav_fname.split('-')[0]
            label = index_label(label_str, binary_class)
        full_fname = os.path.join(wav_dir, wav_fname)
        feature = extract(full_fname, frame_context=frame_context)
        labels = np.append(labels, [label] * len(feature))
        if features is None:
            features = np.empty((0, feature.shape[1]))
        features = np.vstack([features, feature])
    data = (np.array(features), np.array(labels, dtype=np.int))
    if persist:
        import datetime
        timestamp = datetime.datetime.today().strftime('%Y%m%d-%H%M')
        np.savez(f'_models/{timestamp}.features.{features.shape[1]}d', xs=data[0], ys=data[1])
    return data
