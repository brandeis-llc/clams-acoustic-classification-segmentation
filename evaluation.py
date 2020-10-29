import os
import re
import sys
import subprocess
from io import open
from subprocess import PIPE
from datetime import datetime

from bs4 import BeautifulSoup as bs

import numpy as np

def validate_paths(args):
    if (len(args) < 4) or (len(args) > 5):
        # break, explain inputs
        print("To run evaluation.py, please provide\n1) the path to the LDC text and sph files\n2) the path to the audiosegmenter's run.py\n3) the path to the audiosegmenter model and type\n as arguments.")
        quit()
    if os.path.exists(args[1]) is False:
        # break, explain bad file location
        print("Bad LDC file path provided.")
        quit()
    if os.path.exists(args[2]) is False:
        # break, explain bad run location
        print("Bad audiosegmenter run.py path provided.")
        quit()
    if os.path.exists(args[3]) is False:
        # break, explain bad model location
        print("Bad audiosegmenter model path provided.")
        quit()

    return args[1], args[2], args[3]


def parse_ldc(path):
    ldc_unannot = {}
    ldc_outputs = {}

    for file in os.listdir(path):
        if file.endswith(".txt"):
            # parse the annotated file
            ldc_path = os.path.join(path, file)

            unannotated_boundaries, speech_boundaries = read_hub4_annotation(ldc_path)

            # account for unannotated portions at the start of the file
            if len(unannotated_boundaries) == 0 or float(unannotated_boundaries[0][0]) > float(speech_boundaries[0][0]):
                unannotated_boundaries.insert(0, (0.0, speech_boundaries[0]))

            ldc_unannot[file[:-4]] = unannotated_boundaries
            ldc_outputs[file[:-4]] = speech_boundaries

    if len(ldc_outputs) == 0:
        # break, no valid txt files found
        print("No valid txt files found in LDC file path")
        quit()

    return ldc_unannot, ldc_outputs


def read_hub4_annotation(annotation_fname):
    segmentation = {'filename': "", 'speech': [], 'unannotated': []}
    with open(annotation_fname) as annotation:
        tree = bs(annotation, 'lxml')
        episode = tree.find('episode')
        segmentation['filename'] = episode['filename']
        for section in tree.find_all('section'):
            # according to the guidelines, filler and sports_repost sections are
            # not transcribe - and they should not have any 'segment' tags
            if section['type'].lower() == 'filler' or \
                    section['type'].lower() == 'sports_report':
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
    # 1 = nonspeech
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
    return a


def run_audiosegmenter(file_path, run_path, model_path):
    run_prog = os.path.join(run_path, "run.py")

    if(os.path.exists(run_prog) is False):
        print("Run.py not found in run path")
        quit()

    output_path = os.path.join(file_path, "segmented.tsv")
    file = open(output_path, "w+")
    subprocess.call(["python3", run_prog, "-s", model_path, file_path], stdout=file)
    file.close()
    os.chmod(output_path, 0o777)


def parse_audiosegmenter(path, ldc_unannot, ldc_outputs):
    output_path = os.path.join(path, "segmented.tsv")
    file = open(output_path, encoding='utf-8')
    contents = file.read().split("\n")
    file.close()

    error = {}
    sum_of_errors = 0
    sum_of_precisions = 0
    sum_of_recalls = 0
    num_of_files = 0
    sum_of_file_lengths = 0.0
    print('%10s' % 'FILE NAME' + '%10s' % 'ERROR' + '%14s' % 'PRECISION' + '%8s' % 'RECALL')
    for line in contents:
        file = re.findall(r"/([\w\d]+).wav", line)
        if file != []:
            regex_times = re.compile(r"([\d]+[.][\d]+)(\s)")
            audioseg_numbers = [x[0] for x in regex_times.findall(line)]
            ldc_numbers = ldc_outputs[file[0]]
            unannot_numbers = ldc_unannot[file[0]]
            
            miss, false, total_pos, guessed_pos, length = find_miss_false_and_total(audioseg_numbers, ldc_numbers, regex_times)

            fixer = find_fixer(audioseg_numbers, unannot_numbers, regex_times, float(length))

            sum_of_file_lengths = sum_of_file_lengths + float(length) - fixer
            error_val = ((miss + false - fixer) / total_pos) * 100
            precision = (total_pos - miss) / (guessed_pos - fixer)
            recall = (total_pos - miss) / total_pos
            sum_of_errors = sum_of_errors + error_val
            sum_of_precisions = sum_of_precisions + precision
            sum_of_recalls = sum_of_recalls + recall
            num_of_files += 1
            error[file[0]] = file[0] + '\t' + '%.2f' % error_val + "%\t" + '%.2f' % precision + '\t' + '%.2f' % recall
            print('%10s' % file[0] + '%10.2f' % error_val + "%" + '%10.2f' % precision + '%10.2f' % recall)

    average_error = sum_of_errors / num_of_files
    average_precision = sum_of_precisions / num_of_files
    average_recall = sum_of_recalls / num_of_files
    average = '\t' + '%.2f' % average_error + '%\t' + '%.2f' %  average_precision + '\t' + '%.2f' %  average_recall
    print('\nAVERAGE:' + average)
    print('Total length of annotated files:', '%.2f' % ((sum_of_file_lengths / 60) / 60), 'hours')

    if len(error) != len(ldc_outputs):
        print()
        print("Audiosegmenter did not process all of the LDC files successfully.")
        print("Audiosegmenter files processed: ", len(error))
        print("LDC files processed: ", len(ldc_outputs))

    return error, average


def find_fixer(audioseg_numbers, unannot_numbers, regex_times, stop_time):
    # calculate how much audiosegmenter time should be ignored due to coming from unannotated sections
    a = 0
    b = 0
    fixer = 0.0
    
    while a < len(audioseg_numbers) and b < len(unannot_numbers) and float(unannot_numbers[b]) < stop_time:
        as_start = float(audioseg_numbers[a])
        as_end = float(audioseg_numbers[a + 1])
        unannot_start = float(unannot_numbers[b])
        unannot_end = float(unannot_numbers[b + 1])

        if unannot_start == unannot_end:
            # done to account for potential 0.0 - 0.0 entry from unannotated file start correction in parse_ldc
            b += 2
        elif as_end < unannot_start:
            a += 2
        elif unannot_end < as_start:
            b += 2
        else:
            if (as_start <= unannot_start) and (as_end <= unannot_end):
                fixer = fixer + (as_end - unannot_start)
                a += 2
            elif (as_start < unannot_start) and (as_end > unannot_end):
                fixer = fixer + (unannot_end - unannot_start)
                b += 2
            elif (as_start > unannot_start) and (as_end < unannot_end):
                fixer = fixer + (as_end - as_start)
                a += 2
            elif (as_start >= unannot_start) and (as_end >= unannot_end):
                fixer = fixer + (unannot_end - as_start)
                b += 2
    return fixer


def find_miss_false_and_total(audioseg_numbers, ldc_numbers, regex_times):
    x = 0
    y = 0
    false = 0.0
    miss = 0.0
    total_pos = 0.0
    guessed_pos = 0.0
    while x < len(audioseg_numbers) and y < len(ldc_numbers):
        as_start = float(audioseg_numbers[x])
        as_end = float(audioseg_numbers[x + 1])
        ldc_start = float(ldc_numbers[y])
        ldc_end = float(ldc_numbers[y + 1])

        if (as_start > ldc_end):
            miss = miss + (ldc_end - ldc_start)
            total_pos = total_pos + (ldc_end - ldc_start)
            y += 2
        elif (ldc_start > as_end):
            false = false + (as_end - as_start)
            guessed_pos = guessed_pos + (as_end - as_start)
            x += 2
        else:
            start = ldc_start - as_start

            # calculate the amount the start of the speech segment is off by
            if (start < 0):
                miss = miss + (start * -1)
            else:
                false = false + start

            while((((x + 2) < len(audioseg_numbers)) and (float(audioseg_numbers[x + 2]) < ldc_end)) or (((y + 2) < len(ldc_numbers)) and (float(ldc_numbers[y + 2]) < as_end))):
                # check if the next audiosegmenter piece also overlaps with this piece
                if ((x + 2) < len(audioseg_numbers)) and (float(audioseg_numbers[x + 2]) < ldc_end):
                    x += 2
                    guessed_pos = guessed_pos + (as_end - as_start)
                    miss = miss + (float(audioseg_numbers[x]) - as_end)
                    as_start = float(audioseg_numbers[x])
                    as_end = float(audioseg_numbers[x + 1])

                # check if the next ldc piece also overlaps with this piece
                if((y + 2) < len(ldc_numbers)) and (float(ldc_numbers[y + 2]) < as_end):
                    y += 2
                    total_pos = total_pos + (ldc_end - ldc_start)
                    false = false + (float(ldc_numbers[y]) - ldc_end)
                    ldc_start = float(ldc_numbers[y])
                    ldc_end = float(ldc_numbers[y + 1])

            end = ldc_end - as_end
            if (end < 0):
                false = false + (end * -1)
            else:
                miss = miss + end

            guessed_pos = guessed_pos + (as_end - as_start)
            total_pos = total_pos + (ldc_end - ldc_start)
            x += 2
            y += 2

    # do not check for remaining non-speaking sections, as multiple minutes of unannotated (but caught by the segmenter) commercials are often at the end of the file

    while y < len(ldc_numbers):
        # track all of the speaking sections that were missed
        ldc_start = float(ldc_numbers[y])
        ldc_end = float(ldc_numbers[y + 1])
        miss = miss + (ldc_end - ldc_start)
        total_pos = total_pos + (ldc_end - ldc_start)
        y += 2

    return miss, false, total_pos, guessed_pos, ldc_numbers[y-1]


def save_output(error, average):
    keys = error.keys()
    dateTimeObj = datetime.now()
    filename = "evaluation" + dateTimeObj.strftime("%d-%b-%Y_%H.%M.%S") + ".tsv"
    output = open(filename, "w")
    output.write("FILE NAME\tERROR\tPRECISION\tRECALL\n")
    for key in keys:
        output.write(error[key] + "\n")
    final = "AVERAGE" + average
    output.write(final)
    output.close()


if __name__ == '__main__':
    # validate paths
    file_path, run_path, model_path = validate_paths(sys.argv)

    # parse annotated files for output and unannotated sections
    ldc_unannot, ldc_outputs = parse_ldc(file_path)

    # convert sph files to wav files if not already done
    run_sox(file_path)

    # run audiosegmenter
    run_audiosegmenter(file_path, run_path, model_path)

    # parse audiosegmenter output and calculate the error for each relevant file
    error, average = parse_audiosegmenter(file_path, ldc_unannot, ldc_outputs)

    # save output as a file
    save_output(error, average)
