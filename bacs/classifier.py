import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import bacs

BATCH_SIZE = 1024
RANDOM_SEED = 123
LEARNING_RATE = 0.001
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def train_pipeline(X: np.ndarray, Y: np.ndarray):
    tr_ds, te_ds, num_cats = prep_data_pipeline(X, Y, downsample=True)
    model = train(tr_ds, num_cats)
    test(model, te_ds)
    return persist_model(model, '_models')


def predict_pipeline(audio_fpath, model, raw_prob=False):
    from bacs import feature
    import os
    if type(audio_fpath) != str:
        audio_fpath = os.path.join(*audio_fpath)
    feats = feature.extract(audio_fpath)
    predictions = model.predict(feats)
    if not raw_prob:
        predictions = np.argmax(predictions, axis=1)
    return predictions


def prep_data_pipeline(X, Y, downsample=False):
    # current implementation only considers binary classification (speech vs. nonspeech)
    negs = np.where(Y != 0)[0]
    poss = np.where(Y == 0)[0]
    if downsample:
        # we know for sure that negative examples (nonspeech) are much smaller than the positives, so trim positives instances
        np.random.shuffle(poss)
        poss = poss[:len(negs)]

    # because both poss and negs are 1d array, should use hstack to concat them
    data_idxs = np.hstack((poss, negs))
    X_tr, X_te, Y_tr, Y_te = train_test_split(X[data_idxs], Y[data_idxs], test_size=0.1, shuffle=True)
    (traind, num_cats), (testd, _) = to_tf_dataset(X_tr, Y_tr), to_tf_dataset(X_te, Y_te)
    return traind, testd, num_cats


def to_tf_dataset(X, Y):
    Y_onehot = tf.keras.utils.to_categorical(Y, dtype='int16')
    num_cats = Y_onehot.shape[1]
    ds = tf.data.Dataset.from_tensor_slices((X, Y_onehot)).batch(BATCH_SIZE)
    return ds, num_cats


def train(dataset, num_cats):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=30, activation='sigmoid'),
        tf.keras.layers.Dense(units=20, activation='sigmoid'),
        tf.keras.layers.Dense(units=10, activation='sigmoid'),
        tf.keras.layers.Dense(units=num_cats, activation='softmax'),
    ])
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    if num_cats < 2:
        raise ValueError("Number of acoustic categories must be more than one.")
    elif num_cats == 2:
        loss_fn = tf.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
    model.fit(dataset, epochs=20)
    return model


def test(model, dataset):
    model.evaluate(dataset, verbose=2)


def predict(model, data):
    return model.predict(data)


def persist_model(model, persist_dir):
    import datetime
    import os
    timestamp = datetime.datetime.today().strftime('%Y%m%d-%H%M')
    model_path = os.path.join(persist_dir, timestamp)
    model.save(model_path, save_format='tf')
    return model_path


def load_model(model_path=bacs.defmodel_path):
    return tf.keras.models.load_model(model_path)
