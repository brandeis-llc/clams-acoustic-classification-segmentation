import numpy as np

def smooth(predictions: np.ndarray):
    NotImplementedError


def frames_to_durations(predictions):
    # assumes frame size to be a hundredth second (10ms)
    # smoothed = smooth(predictions)
    smoothed = predictions
    speech = False
    segments = {}
    cur_speech_segment_started = 0
    for f_num, frame in enumerate(smoothed):
        if speech and frame == 1:
            segments[cur_speech_segment_started] = f_num-1
            speech = False
        elif not speech and frame == 0:
            cur_speech_segment_started = f_num
            speech = True
    for start, end in segments.items():
        print(f'{start / 100}\t{end/100}', end='\t')
    print('\n')


