import reader, feature, classifier, smoothing, writer
import sys
import os

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        '-d', '--download',
        action='store_true',
        help='Flag to download input datasets; train and test sets for classifier as well as train set for word embedding.'
    )

    parser.add_argument(
        '-t', '--train',
        default='',
        action='store',
        nargs='?',
        help='Flag to invoke training pipeline. Use an argument to pass directory of training data. '
    )
    parser.add_argument(
        '-s', '--segment',
        default='',
        action='store',
        nargs=2,
        help='Flag to invoke segmentation pipeline. First arg to specify model path, and second to specify directory where wave files are. '
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

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.train:
        X, Y = feature.extract_all(reader.read_wavs(args.train), train=True, binary_class=True)
        model_path = classifier.train_pipeline(X, Y)
        print("============")
        print("model saved at " + model_path)
        print("============")

    if args.segment:
        for wav in reader.read_wavs(args.segment[1], file_ext=['mp3', 'wav']):
            model = classifier.load_model(args.segment[0])
            predicted = classifier.predict_pipeline(wav, model)
            smoothed = smoothing.smooth(predicted)
            speech_portions, total_frames = writer.index_frames(smoothed)
            audio_fname = os.path.join(*wav)
            writer.print_durations(speech_portions, audio_fname, total_frames)
            if args.out:
                print('writing files')
                writer.slice_speech(speech_portions, audio_fname)


