import os
import re
import sys
from io import open


def validate_paths(args):
    if len(args) != 4:
        # break, explain inputs
        print("To run evaluation.py, please provide\n1) the path to the LDC text and sph files\n2) the path to the audiosegmenter's run.py\n3) the path to the audiosegmenter model including the file name and type\n as arguments.")
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
            ldc_file = open(ldc_path, encoding='utf-8')
            contents = ldc_file.read()
            ldc_file.close()

            section_boundaries = find_ldc_unannot(contents)
            speech_boundaries = find_ldc_output(contents)

            ldc_unannot[file[:-4]] = section_boundaries
            ldc_outputs[file[:-4]] = speech_boundaries

    if len(ldc_outputs) == 0:
        # break, no valid txt files found
        print("No valid txt files found in LDC file path")
        quit()

    return ldc_unannot, ldc_outputs


def find_ldc_unannot(contents):
    # find all sections
    regex_unannot = re.compile(r"Section   Type=([^>]*)")
    unannot = regex_unannot.findall(contents)

    # get the start and stop times of unannotated sections
    section_boundaries = []
    for section in unannot:
        if (section[:6] != "Story ") and (section[:6] != "Filler"):
            regex_times = re.compile(r"(_time=)(\d*.\d*)")
            times = regex_times.findall(section)
            section_boundaries.append(times[0][1])
            section_boundaries.append(times[1][1])

    return section_boundaries


def find_ldc_output(contents):
    # find the spoken segments
    regex_segs = re.compile(r"Segment [^>]*")
    segments = regex_segs.findall(contents)

    # find the start and stop times in the segments
    regex_times = re.compile(r"(_time=)(\d*.\d*)")
    start = regex_times.findall(segments[0])[0][1]
    end = start
    speech_boundaries = []
    for segment in segments:
        times = regex_times.findall(segment)
        if int(float(times[0][1])) <= (int(float(end)) + 3):
            end = times[1][1]
        else:
            speech_boundaries.append('%.2f' % (float(start)))
            speech_boundaries.append('%.2f' % (float(end)))
            start = times[0][1]
            end = times[1][1]
    speech_boundaries.append('%.2f' % (float(start)))
    speech_boundaries.append('%.2f' % (float(end)))

    return speech_boundaries


def run_sox(file_path):
    for file in os.listdir(file_path):
        if file.endswith(".sph") and (os.path.exists(os.path.join(file_path, (file[:-3] + "wav"))) is False):
            os.system("sox " + file + " " + file[:-3] + "wav")


def run_audiosegmenter(file_path, run_path, model_path):
    run_prog = os.path.join(run_path, "run.py")

    if(os.path.exists(run_prog) is False):
        print("Run.py not found in run path")
        quit()

    print(run_prog + " -s " + model_path + " " + file_path)
    #os.system(run_prog + " -s " + model_path + " " + file_path)


def parse_audiosegmenter(path, ldc_unannot, ldc_outputs):
    output_path = os.path.join(path, "segmented.tsv")
    file = open(output_path, encoding='utf-8')
    contents = file.read().split("\n")
    file.close()

    error = {}
    for line in contents:
        file = re.findall(r"/([\w\d]+).wav", line)
        if file != []:
            regex_times = re.compile(r"([\d]+[.][\d]+)(\s)")
            audioseg_numbers = [x[0] for x in regex_times.findall(line)]
            ldc_numbers = ldc_outputs[file[0]]
            unannot_numbers = ldc_unannot[file[0]]

            fixer = find_fixer(audioseg_numbers, unannot_numbers, regex_times)
            miss, false, total = find_miss_false_and_total(audioseg_numbers, ldc_numbers, regex_times)

            error_val = ((miss + false - fixer) / total) * 100
            error[file[0]] = file[0] + ' \t ' + '%.2f' % error_val + "%"
            print(file[0], 'error:', '%.2f' % error_val, "%")

    if len(error) != len(ldc_outputs):
        print("Audiosegmenter did not process all of the LDC files successfully.")
        print("Audiosegmenter files processed: ", len(error))
        print("LDC files processed: ", len(ldc_outputs))

    return error


def find_fixer(audioseg_numbers, unannot_numbers, regex_times):
    # calculate how much audiosegmenter time should be ignored due to coming from unannotated sections
    a = 0
    b = 0
    fixer = 0.0
    while a < len(audioseg_numbers) and b < len(unannot_numbers):
        as_start = float(audioseg_numbers[a])
        as_end = float(audioseg_numbers[a + 1])
        unannot_start = float(unannot_numbers[b])
        unannot_end = float(unannot_numbers[b + 1])

        if as_end < unannot_start:
            a += 2
        elif unannot_end < as_start:
            b += 2
        else:
            if (as_start < unannot_start) and (as_end < unannot_end):
                fixer = fixer + (as_end - unannot_start)
                a += 2
            elif (as_start < unannot_start) and (as_end > unannot_end):
                fixer = fixer + (unannot_end - unannot_start)
                b += 2
            elif (as_start > unannot_start) and (as_end < unannot_end):
                fixer = fixer + (as_end - as_start)
                a += 2
            elif (as_start > unannot_start) and (as_end > unannot_end):
                fixer = fixer + (unannot_end - as_start)
                b += 2

    return fixer


def find_miss_false_and_total(audioseg_numbers, ldc_numbers, regex_times):
    x = 0
    y = 0
    false = 0.0
    miss = 0.0
    total = 0.0
    while x < len(audioseg_numbers) and y < len(ldc_numbers):
        as_start = float(audioseg_numbers[x])
        as_end = float(audioseg_numbers[x + 1])
        ldc_start = float(ldc_numbers[y])
        ldc_end = float(ldc_numbers[y + 1])

        if (as_start > ldc_end):
            miss = miss + (ldc_end - ldc_start)
            total = total + (ldc_end - ldc_start)
            y += 2
        elif (ldc_start > as_end):
            false = false + (as_end - as_start)
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
                    miss = miss + (float(audioseg_numbers[x]) - as_end)
                    as_start = float(audioseg_numbers[x])
                    as_end = float(audioseg_numbers[x + 1])

                # check if the next ldc piece also overlaps with this piece
                if((y + 2) < len(ldc_numbers)) and (float(ldc_numbers[y + 2]) < as_end):
                    y += 2
                    total = total + (ldc_end - ldc_start)
                    false = false + (float(ldc_numbers[y]) - ldc_end)
                    ldc_start = float(ldc_numbers[y])
                    ldc_end = float(ldc_numbers[y + 1])

            end = ldc_end - as_end
            if (end < 0):
                false = false + (end * -1)
            else:
                miss = miss + end

            total = total + (ldc_end - ldc_start)
            x += 2
            y += 2

    while x < len(audioseg_numbers):
        # track all of the non-speaking sections that were mislabelled
        as_atart = float(audioseg_numbers[x])
        as_end = float(audioseg_numbers[x + 1])
        false = false + (as_end - as_start)
        x += 2
    while y < len(ldc_numbers):
        # track all of the speaking sections that were missed
        ldc_start = float(ldc_numbers[y])
        ldc_end = float(ldc_numbers[y + 1])
        miss = miss + (ldc_end - ldc_start)
        total = total + (ldc_end - ldc_start)
        y += 2

    return miss, false, total


def save_output(error):
    keys = error.keys()

    output = open("evaluation.tsv", "w")
    for key in keys:
        output.write(error[key] + "\n")
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
    error = parse_audiosegmenter(file_path, ldc_unannot, ldc_outputs)

    # save output as a file
    save_output(error)
