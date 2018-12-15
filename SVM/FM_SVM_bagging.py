# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager


import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import svm
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))

def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter).values.squeeze()


def build_rating_matrix(user_movie_rating_triplets):
    """
    Create the rating matrix from triplets of user and movie and ratings.

    A rating matrix `R` is such that `R[u, m]` is the rating given by user `u`
    for movie `m`. If no such rating exists, `R[u, m] = 0`.

    Parameters
    ----------
    user_movie_rating_triplets: array [n_triplets, 3]
        an array of trpilets: the user id, the movie id, and the corresponding
        rating.
        if `u, m, r = user_movie_rating_triplets[i]` then `R[u, m] = r`

    Return
    ------
    R: sparse csr matrix [n_users, n_movies]
        The rating matrix
    """
    rows = user_movie_rating_triplets[:, 0]
    cols = user_movie_rating_triplets[:, 1]
    training_ratings = user_movie_rating_triplets[:, 2]

    return sparse.coo_matrix((training_ratings, (rows, cols))).tocsr()

def slice_feature(data_matrix, n):
    return data_matrix[:, n]

def create_learning_matrices(rating_matrix, user_movie_pairs):
    """
    Create the learning matrix `X` from the `rating_matrix`.

    If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
    corresponding to user `u` and movie `m`. The feature vector is composed
    of `n_users + n_movies` features. The `n_users` first features is the
    `u-th` row of the `rating_matrix`. The `n_movies` last features is the
    `m-th` columns of the `rating_matrix`

    In other words, the feature vector for a pair (user, movie) is the
    concatenation of the rating the given user made for all the movies and
    the rating the given movie receive from all the user.

    Parameters
    ----------
    rating_matrix: sparse matrix [n_users, n_movies]
        The rating matrix. i.e. `rating_matrix[u, m]` is the rating given
        by the user `u` for the movie `m`. If the user did not give a rating for
        that movie, `rating_matrix[u, m] = 0`
    user_movie_pairs: array [n_predictions, 2]
        If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
        must relate to user `u` and movie `m`

    Return
    ------
    X: sparse array [n_predictions, n_users + n_movies]
        The learning matrix in csr sparse format
    """
    # Feature for users
    prefix = 'data/'
    data_user = load_from_csv(os.path.join(prefix, 'data_user.csv'))

    user_features = rating_matrix[user_movie_pairs[:, 0]]

    # Features for movies
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()

    # Feature age
    age = slice_feature(data_user, 1)

    age_stack = np.zeros((len(user_movie_pairs), 1))
    for i in np.arange(len(user_movie_pairs)):
        age_stack[i] = age[user_movie_pairs[i, 0] - 1]

    X = np.hstack((user_features, movie_features))

    return X


def make_submission(y_predict, user_movie_ids, file_name='submission',
                    date=True):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predict: array [n_predictions]
        The predictions to write in the file. `y_predict[i]` refer to the
        user `user_ids[i]` and movie `movie_ids[i]`
    user_movie_ids: array [n_predictions, 2]
        if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
        for user `u` and movie `m`
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """
    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Naming the file
    if date:
        file_name = '{}/{}_{}'.format(directory, file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for (user_id, movie_id), prediction in zip(user_movie_ids,
                                                 y_predict):

            if np.isnan(prediction):
                raise ValueError('The prediction cannot be NaN')
            line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
            handle.write(line)
    return file_name


if __name__ == '__main__':
    prefix = 'data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    user_movie_rating_triplets = np.hstack((training_user_movie_pairs,
                                            training_labels.reshape((-1, 1))))

    rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    print("Load reconstructed..")
    reconstructed = np.loadtxt('reconstructed/reconstructed_mat_10_0002_001_2000.txt')

    row, col = rating_matrix.nonzero()
    array_rating = rating_matrix.toarray()

    print("Correction and remplacement of reconstructed..")
    # Correction of the values <1 and >5
    for i in range(0,912):
        for j in range(0,1542):
            if reconstructed[i][j] < 1.0:
                reconstructed[i][j] = 1.0
            if reconstructed[i][j] > 5.0:
                reconstructed[i][j] = 5.0

    for i, j in zip(row, col):
        reconstructed[i][j] = array_rating[i][j]

    X_ls = create_learning_matrices(reconstructed, training_user_movie_pairs)
    y_ls = training_labels

    start = time.time()
    model = svm.SVR(kernel = 'poly', degree = 2, C = 0.7)
    "boostedModel = AdaBoostRegressor(base_estimator= model, n_estimators = 10, loss = 'square')"
    baggedModel = BaggingRegressor(base_estimator=model, n_estimators = 50, max_features = 0.6, max_samples = 0.5, bootstrap = False)

    X_train, X_test, y_train, y_test = train_test_split(X_ls, y_ls, test_size=0.30)

    with measure_time('Training'):
        print('Training...')
        baggedModel.fit(X_train, y_train)

    y_pred = baggedModel.predict(X_test)
    print('MSE on 0.2 of LS: ',mean_squared_error(y_true, y_pred, multioutput='uniform_average'))

    with measure_time('Training'):
        print('Training...')
        baggedModel.fit(X_ls, y_ls)

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = create_learning_matrices(rating_matrix, test_user_movie_pairs)

    # Predict
    y_pred = baggedModel.predict(X_ts)
    # y_pred = model.predict(X_ts)
    i=0
    while i<len(y_pred):
        "y_pred[i] = round(y_pred[i])"
        if y_pred[i] > 5.0:
            y_pred[i] = 5.0
        if y_pred[i] < 1.0:
            y_pred[i] = 1.0
        i = i+1


    # Making the submission file
    fname = make_submission(y_pred, test_user_movie_pairs, 'regression')
    print('Submission file "{}" successfully written'.format(fname))
