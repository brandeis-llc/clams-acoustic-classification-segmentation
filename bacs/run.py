#! /usr/bin/env python3

import sys
import os
import bacs


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        '-d', '--download',
        action='store_true',
        help='Flag to download input datasets; train and test sets for '
             'classifier as well as train set for word embedding.'
    )

    parser.add_argument(
        '-t', '--train',
        default='',
        action='store',
        nargs='?',
        help='Flag to invoke training pipeline. Must pass an argument that '
             'points to the training data. If the arg is a directory, training '
             'features will be extracted from all wav files in the directory - '
             'extracted features will be stored as a npz file in _model '
             'directory for future uses. If the arg is a npz file, training '
             'features stored in the file will be unpacked and be used.'
    )
    parser.add_argument(
        '-s', '--segment',
        default='',
        action='store',
        metavar='AUDIO',
        help='Flag to invoke segmentation pipeline. '
             'First arg to specify model '
             'path, and second to specify directory where wave files are. '
    )
    parser.add_argument(
        '-e', '--evaluate',
        default='',
        action='store',
        metavar='HUB4',
        help='Run evaluation against HUB4 annotation. '
             'HUB4 annotations (txt) and audio files (sph) must be located '\
             'in '
             'a single directory. '
             'Use with --model to specify the model to evaluate. '
    )
    parser.add_argument(
        '-m', '--model',
        default=bacs.defmodel_path,
        action='store',
        help='Specify model to use for --segment or --evaluate. '
             'The argument must be a directory name where tensorflow:SavedModel is stored. '
    )
    parser.add_argument(
        '-o', '--out',
        default='',
        action='store_true',
        help='Only valid with \'segment\' flag. When given, new wav files are '
             'generated from an input audio file, each stores a single \'speech\' '
             'segment. Newly generated files are stored in a subdirectory named '
             'after the full audio file, and suffixed with starting position '
             'in seconds (to two decimal places).'
    )
    parser.add_argument(
        '-n', '--numfiles',
        default=sys.maxsize,
        action='store',
        type=int,
        help='Valid with \'segment\' and \'evaluate\' flags. When given, '
             'the number of files in data directory to process or evaluate will '
             'be limited to the given value.'
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    from bacs import reader, feature, classifier, smoothing, writer
    if args.train:
        if args.train.endswith('.npz'):
            import numpy
            npzarrays = numpy.load(args.train)
            X, Y = npzarrays['xs'], npzarrays['ys']
        else:
            X, Y = feature.extract_all(reader.read_audios(args.train), train=True, binary_class=True, persist=True)
        model_path = classifier.train_pipeline(X, Y)
        print("============")
        print("model saved at " + model_path)
        print("============")

    if args.segment:
        model = classifier.load_model(args.model)
        for wav in reader.read_audios(args.segment, file_per_dir=args.numfiles):
            predicted = classifier.predict_pipeline(wav, model)
            smoothed = smoothing.smooth(predicted)
            speech_portions, total_frames = writer.index_frames(smoothed)
            audio_fname = os.path.join(*wav)
            writer.print_durations(speech_portions, audio_fname, total_frames)
            if args.out:
                print('writing files')
                writer.slice_speech(speech_portions, audio_fname)

    if args.evaluate:
        from bacs import evaluation
        model = classifier.load_model(args.model)
        evaluation.evaluate_files(args.evaluate, model, args.numfiles)


if __name__ == '__main__':
    main()
