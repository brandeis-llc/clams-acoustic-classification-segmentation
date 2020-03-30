import numpy as np


def minimum_change_support(predictions: np.ndarray, minimum_window_size=300):
    for i in range(1, len(predictions)):
        cur_label = predictions[i]
        if cur_label != 0 and np.sum(predictions[max(0, i - minimum_window_size)] == cur_label) < (minimum_window_size // 2):
            predictions[i] = predictions[i-1]


def mode_smooth(predictions: np.ndarray, smooth_window=20):
    from scipy import stats
    for i in range(len(predictions)):
        s = max(0, i-smooth_window)
        e = min(len(predictions), i+smooth_window)
        # print(predictions[s:e])
        predictions[i] = stats.mode(predictions[s:e])[0]


def frames_to_durations(predictions):
    # assumes frame size to be a hundredth second (10ms)
    # smoothings happen in-place
    mode_smooth(predictions)
    minimum_change_support(predictions)
    speech = False
    segments = {}
    cur_speech_segment_started = 0
    for f_num, frame in enumerate(predictions):
        if speech and frame == 1:
            segments[cur_speech_segment_started] = f_num-1
            speech = False
        elif not speech and frame == 0:
            cur_speech_segment_started = f_num
            speech = True
    if speech:
        segments[cur_speech_segment_started] = len(predictions)-1

    for start, end in segments.items():
        print(f'{start / 100}\t{end/100}', end='\t')
    print('\n')


