import reader, feature, classifier
import sys

if __name__ == '__main__':

    X, Y = feature.extract_all(reader.read_wavs(sys.argv[1], file_per_dir=2), train=True, binary_class=True)
    model_path = classifier.train_pipeline(X, Y)
    classifier.predict_pipeline(X, model_path)
