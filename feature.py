import os
import warnings

import librosa
import numpy as np

labels = {'speech': 0, 'music': 1, 'noise': 2}
FRAME_SIZE = 10 # milliseconds
CONTEXT_FRAMES = 0
ZCR=False
MFCC_NUM=40


def extract(wav_fname, frame_size=FRAME_SIZE, context_frames=CONTEXT_FRAMES, zcr=ZCR, mfcc_num=MFCC_NUM, verbose=False, **kwargs):
    # will sample 16000 points per second
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = librosa.load(wav_fname, sr=16000, **kwargs)
    if verbose:
        print(f'feature extracting: {wav_fname}\t', end='', flush=True)
    feats = spectral_feats(audio, frame_size, sr, mfcc_num, zcr)
    if context_frames > 0:
        if verbose:
            print(f'(normalizing)\t', end='', flush=True)
        feats = temporal_feats(feats, context_frames)
    if verbose:
        print(f'length: {librosa.get_duration(audio, sr)} secs, frames: {feats.shape}', end='', flush=True)
        print('', flush=True)
    return feats


def spectral_feats(audio, frame_size, samplerate, mfcc_num, zcr):
    frame_sliding_size = samplerate // (1000 // frame_size)
    feats = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=mfcc_num, hop_length=frame_sliding_size)
    if zcr:
        zcrs = librosa.feature.zero_crossing_rate(y=audio, hop_length=frame_sliding_size)
        feats = np.concatenate((feats, zcrs), axis=0)
    # transpose so that rows are time frames
    return feats.T


def temporal_feats(spectral_feats, context_frames):
    last_frame = len(spectral_feats)
    temporalized_frames = None
    for i in range(last_frame):
        # +1 to the end as array slicing is exclusive
        context = spectral_feats[max(0, i - context_frames):min(last_frame, i + context_frames) + 1]
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


def extract_all(wav_paths, frame_size=FRAME_SIZE, context_frames=CONTEXT_FRAMES, zcr=ZCR, mfcc_num=MFCC_NUM, train=False, binary_class=True, persist=False):
    """

    :param wav_paths: A list of audio files to extract. Must be (parent_dir, audio_filename) tuples.
    :param frame_size: The size of minimal time unit for a spectral feature (should be in milliseconds)
    :param context_frames: Number of adjacent frames to be used for extraction of temporal features. When 0, no temporal features will be used. Note that the context in both directions will be used (2 * N frames).
    :param train: When true, it will try to obtain gold labels from the file name. Otherwise, labels will remain None
    :param binary_class: When true, all non-zero labels are collapsed into 1 and treated as False (0 = True)
    :param persist: When true, store extracted features in to _models directory.
    :return: Two numpy arrays. First is a feature matrix (#frames * #features), seconds is a label array (#frames). #feature = (#mfcc_num + (zcr? 1:0) ) * (context==0? 3:1)
    """
    features = None
    labels = np.empty(0)
    label = None
    for wav_dir, wav_fname in wav_paths:
        if train:
            label_str = wav_fname.split('-')[0]
            label = index_label(label_str, binary_class)
        full_fname = os.path.join(wav_dir, wav_fname)
        feature = extract(full_fname, frame_size=frame_size, context_frames=context_frames, zcr=zcr, mfcc_num=mfcc_num)
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
