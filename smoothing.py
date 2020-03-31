import numpy as np


def minimum_change_support(predictions: np.ndarray, minimum_window_size=300):
    for i in range(1, len(predictions)):
        cur_label = predictions[i]
        minimum_window = predictions[max(0,i - minimum_window_size):i]
        if cur_label != 0 and np.sum(minimum_window == cur_label) < (len(minimum_window) // 2):
            predictions[i] = predictions[i-1]


def mode_smooth(predictions: np.ndarray, smooth_window=20):
    from scipy import stats
    for i in range(len(predictions)):
        s = max(0, i-smooth_window)
        e = min(len(predictions), i+1+smooth_window)
        predictions[i] = stats.mode(predictions[s:e])[0]

def trim_short_noises(predictions: np.ndarray, threshold=300):
    i = 0
    cur = predictions[0]
    while i < len(predictions):
        if predictions[i] == 1:
            next_nonzeros = np.where(predictions[i:] == 0)[0]
            if len(next_nonzeros) == 0: # nore more flips left
                break
            noise_len = next_nonzeros[0]
            #  print(i, noise_len)
            if noise_len < threshold:
                predictions[i:i+noise_len] = 0
            i += noise_len
        else:
            i += 1


def frames_to_durations(predictions, report_ratio=False):
    # assumes frame size to be a hundredth second (10ms)
    # smoothings happen in-place
    #  mode_smooth(predictions)
    #  minimum_change_support(predictions)
    trim_short_noises(predictions)
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
    speech_sum = 0
    for start, end in segments.items():
        print(f'{start/100}\t{end/100}', end='\t')
        speech_sum += (end - start)
        
    if report_ratio:
        print(f"speech_ratio: {(speech_sum / len(predictions)):.2%} ({speech_sum} / {len(predictions)})")
    print('\n')


