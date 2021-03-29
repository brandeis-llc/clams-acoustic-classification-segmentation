import os

import numpy as np
from bs4 import BeautifulSoup as bs
from sklearn import metrics

from bacs import classifier
from bacs import feature
from bacs import smoothing


def read_hub4_annotation(annotation_fname):
    if type(annotation_fname) != str:
        annotation_fname = os.path.join(*annotation_fname)
    segmentation = {'filename': "", 'speech': [], 'unannotated': []}
    with open(annotation_fname) as annotation:
        tree = bs(annotation, 'lxml')
        episode = tree.find('episode')
        segmentation['filename'] = episode['filename']
        for section in tree.find_all('section'):
            # according to the guidelines, filler and sports_repost sections are
            # not transcribe - and they should not have any 'segment' tags
            # also, we found "Local_News" and "Commercial" type sections also do not have segments in most cases 
            # (only 5 out of 118 "Local_News" sections had at least one segment, and 5 out of 553 "Commercial" sections had at least one segment)
            # so we decided to treat local_news & commercial as unannotated, too
            if section['type'].lower() in ('filler', 'commercial', 'local_news', 'sports_report'):
                segmentation['unannotated'].append((float(section['s_time']), float(section['e_time'])))
            for segment in section.find_all('segment'):
                segmentation['speech'].append((float(segment['s_time']), float(segment['e_time'])))
    # account for unannotated portions at the start of the file
    # if len(segmentation['unannotated']) == 0 or segmentation['unannotated'][0][0] > segmentation['speech'][0][0]:
    #     segmentation['unannotated'].insert(0, (0.0, segmentation['speech'][0][0]))

    return segmentation


def to_nparray(segment_dict, audio_duration, frame_size=feature.FRAME_SIZE):
    """
    Converts XML annotation of audio segmentation into a numpy array

    :param audio_duration: duration of the audio in milliseconds
    :param frame_size: size of a "frame" in milliseconds. frame is a time slice of each cell of the array represents.
    :param segment_dict: dictionary where speech segmentation annotation is encoded

    :return:
    """
    # 0 = speech
    # 1 = non-speech
    # -1 = unannotated
    a = np.ones(audio_duration//frame_size)

    def to_frame_num(start_end_tuple):
        return list(map(lambda x: int(x*1000) // frame_size, start_end_tuple))

    for speech_seg in segment_dict['speech']:
        start, end = to_frame_num(speech_seg)
        a[start:end] = 0
    for unannotated_seg in segment_dict['unannotated']:
        start, end = to_frame_num(unannotated_seg)
        a[start:end] = -1

    # smooth out short non-speeches
    # https://github.com/brandeis-llc/acoustic-classification-segmentation/blob/v1/evaluation.py#L94-L96
    smoothing.trim_short_noises(a, 1000 // frame_size) # 3000 ms = 3 seconds

    # account for unannotated portions at the start of the file
    # https://github.com/brandeis-llc/acoustic-classification-segmentation/blob/v1/evaluation.py#L46-L49
    start, _ = to_frame_num(segment_dict['speech'][0])
    a[0:start] = -1

    # do not check for remaining non-speaking sections, as multiple minutes of unannotated (but caught by the segmenter) commercials are often at the end of the file
    # https://github.com/brandeis-llc/acoustic-classification-segmentation/blob/v1/evaluation.py#L273
    _, end = to_frame_num(segment_dict['speech'][-1])
    a[end:] = -1

    print(f'annotation loaded - '
          f'unannotated: {len(np.where(a == -1)[0]) / len(a):.2%}, '
          f'speech: {len(np.where(a == 0)[0]) / len(a):.2%}, '
          f'non-speech: {len(np.where(a == 1)[0]) / len(a):.2%}')
    return a


def p_r_f(hub4_array, predictions):
    """
    predictions must be 1d array of labels, not k-d raw probabilities
    """
    annotated_idx = np.where(hub4_array != -1)[0]
    return metrics.precision_recall_fscore_support(hub4_array[annotated_idx], predictions[annotated_idx], pos_label=1, average='binary')


def roc(hub4_array, predictions):
    annotated_idx = np.where(hub4_array != -1)[0]
    return metrics.roc_curve(hub4_array[annotated_idx], predictions[annotated_idx][:,0], pos_label=1)


def evaluate_file(sph_fname, txt_fname, classifier_model):
    probs = classifier.predict_pipeline(sph_fname, classifier_model, raw_prob=True)
    duration = probs.shape[0] * feature.FRAME_SIZE # number of frames * frame size
    y_hats = np.argmax(probs, axis=1)
    y_hats = smoothing.smooth(y_hats)
    ys = to_nparray(read_hub4_annotation(txt_fname), duration)
    return probs, y_hats, ys, p_r_f(ys, y_hats)


def evaluate_files(hub4_dir, model, numfiles):
    import reader
    all_probabilities = np.empty((0,2))
    all_predictions = np.empty((0,))
    all_annotations = np.empty((0,))
    for sph_path in reader.read_audios(hub4_dir, file_ext=['sph'], file_per_dir=numfiles):
        base_fname = os.path.splitext(sph_path[1])[0]
        probs, predictions, annotations, scores = evaluate_file(os.path.join(*sph_path), os.path.join(hub4_dir, base_fname + '.txt'), model)
        all_probabilities = np.vstack((all_probabilities, probs))
        all_predictions = np.hstack((all_predictions, predictions))
        all_annotations = np.hstack((all_annotations, annotations))
        print(sph_path[1], scores, flush=True)
    print('TOTAL', p_r_f(all_annotations, all_predictions), flush=True)

