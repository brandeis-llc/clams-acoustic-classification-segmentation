import numpy as np


def minimum_change_support(predictions: np.ndarray, minimum_window_size=300):
    for i in range(1, len(predictions)):
        cur_label = predictions[i]
        minimum_window = predictions[max(0, i - minimum_window_size):i]
        if cur_label != 0 and np.sum(minimum_window == cur_label) < (len(minimum_window) // 2):
            predictions[i] = predictions[i - 1]


def mode_smooth(predictions: np.ndarray, smooth_window=20):
    from scipy import stats
    for i in range(len(predictions)):
        s = max(0, i - smooth_window)
        e = min(len(predictions), i + 1 + smooth_window)
        predictions[i] = stats.mode(predictions[s:e])[0]


def trim_short_speech(predictions: np.ndarray, threshold=200):
    i = 0
    while i < len(predictions):
        if predictions[i] == 0:
            next_nonzeros = np.where(predictions[i:] == 1)[0]
            if len(next_nonzeros) == 0:  # nore more flips left
                break
            speech_len = next_nonzeros[0]
            #  print(i, noise_len)
            if speech_len < threshold:
                predictions[i:i + speech_len] = 1
            i += speech_len
        else:
            i += 1


def trim_short_noises(predictions: np.ndarray, threshold=300):
    i = 0
    cur = predictions[0]
    while i < len(predictions):
        if predictions[i] == 1:
            next_speeches = np.where(predictions[i:] == 0)[0]
            if len(next_speeches) == 0:  # nore more flips left
                break
            noise_len = next_speeches[0]
            #  print(i, noise_len)
            if noise_len < threshold:
                predictions[i:i + noise_len] = 0
            i += noise_len
        else:
            i += 1


def smooth(predictions):
    # assumes frame size to be a hundredth second (10ms)
    # smoothings happen in-place
    # mode_smooth(predictions)
    # minimum_change_support(predictions)
    trim_short_noises(predictions, threshold=100)
    trim_short_speech(predictions)
    return predictions
