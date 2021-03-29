import ffmpeg
import os
from bacs import feature

PRECISION = 1000 // feature.FRAME_SIZE


def index_frames(predictions):
    speech = False
    segments = {}
    cur_speech_segment_started = 0
    for f_num, frame in enumerate(predictions):
        if speech and frame == 1:
            segments[cur_speech_segment_started] = f_num - 1
            speech = False
        elif not speech and frame == 0:
            cur_speech_segment_started = f_num
            speech = True
    if speech:
        segments[cur_speech_segment_started] = len(predictions)
    return segments, len(predictions)


def print_durations(indexed_speech_segements, input_audio_fname, total_len=None):
    speech_sum = 0
    print(input_audio_fname, end='\t', flush=True)
    for start, end in indexed_speech_segements.items():
        print(f'{start / PRECISION}\t{end / PRECISION}', end='\t', flush=True)
        speech_sum += (end - start)

    if total_len is not None:
        print(f"speech_ratio: {(speech_sum / total_len):.2%} ({speech_sum} / {total_len})", end='', flush=True)
    print('', flush=True)


def slice_speech(indexed_speech_segements, input_audio_fname):
    output_dirname = input_audio_fname[:-4]
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirname)
    elif not os.path.isdir(output_dirname):
        raise IOError(f'{output_dirname} file exists and thus output directory cannot be created.')
    else:
        for f in os.listdir(output_dirname):
            os.remove(os.path.join(output_dirname, f))

    for start, end in indexed_speech_segements.items():
        start = start / PRECISION
        end = end / PRECISION
        output_fname = f'{output_dirname.split(os.sep)[-1]}.{str(start)}.wav'
        in_stream = ffmpeg.input(input_audio_fname, f=input_audio_fname[-3:], ss=start, t=end-start)
        in_stream.output(os.path.join(output_dirname, output_fname)).run(overwrite_output=True)
