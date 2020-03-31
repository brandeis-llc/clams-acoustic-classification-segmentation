import reader, feature, classifier, smoothing
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
            print(os.path.join(*wav), end='\t')
            smoothing.frames_to_durations(predicted, True)

