import reader, feature, classifier
import sys

if __name__ == '__main__':

    X, Y = feature.extract_all(reader.read_wavs(sys.argv[1]), train=True, binary_class=True)
    classifier.train_pipeline(X, Y)
