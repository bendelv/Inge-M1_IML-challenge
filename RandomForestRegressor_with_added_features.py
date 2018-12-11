# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
from contextlib import contextmanager


import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

    prefix = 'data/'
    data_user = load_from_csv(os.path.join(prefix, 'data_user.csv'))
    "Feature for users"
    # Feature gender
    gender = slice_feature(data_user, 2)

    for i in np.arange(len(gender)):
        if gender[i] == 'M':
            gender[i] = 0
        else:
            gender[i] = 1

    gender_stack = np.zeros((len(user_movie_pairs), 1))
    for i in np.arange(len(user_movie_pairs)):
        gender_stack[i] = gender[user_movie_pairs[i, 0] - 1]

    # Feature age
    age = slice_feature(data_user, 1)

    age_stack = np.zeros((len(user_movie_pairs), 1))
    for i in np.arange(len(user_movie_pairs)):
        age_stack[i] = age[user_movie_pairs[i, 0] - 1]

    # Feature user ratings on movies
    rating_matrix = rating_matrix.tocsr()
    user_features = rating_matrix[user_movie_pairs[:, 0]]
    
    
    # Features for movies
    "data_movie = load_from_csv(os.path.join(prefix, 'data_movie.csv'))"
    "data_movie = pd.read_csv(os.path.join(prefix, 'data_movie.csv'), delimiter=',').values.squeeze()"
    
    data_movie = pd.read_csv(os.path.join(prefix, 'data_movie.csv'), delimiter=',', encoding='latin-1').values.squeeze()
    

    # Feature genre 5 - 23
    genre = data_movie[:, 5:23]
    print(genre.shape, genre[1, :])

    genres_stack = np.zeros((len(user_movie_pairs), genre.shape[1]))
    print(genres_stack.shape)
    for i in np.arange(len(user_movie_pairs)):
        genres_stack[i][:] = genre[user_movie_pairs[i, 1] - 1, :]
    

    #Feature movie rating by users
    rating_matrix = rating_matrix.tocsc()
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()

    X = sparse.hstack((user_features, movie_features))
    X = sparse.hstack((X, gender_stack))
    X = sparse.hstack((X, age_stack))
    "X = sparse.hstack((X, genres_stack))"

    return X.tocsr()


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

    # Build the learning matrix
    rating_matrix = build_rating_matrix(user_movie_rating_triplets)
    X_ls = create_learning_matrices(rating_matrix, training_user_movie_pairs)

    # Build the model
    y_ls = training_labels
    "X_ls, X_ts, y_ls, y_ts = train_test_split(X, y, test_size=0.2)"
    start = time.time()
    model = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=8, 
                                  min_samples_split=2, min_samples_leaf=1, 
                                  min_weight_fraction_leaf=0.0, max_features='auto', 
                                  max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, bootstrap=True, oob_score=False, 
                                  n_jobs=-1, random_state=None, verbose=0, warm_start=False)
    
    
  
    scores = cross_val_score(model, X_ls, y_ls, scoring= 'neg_mean_squared_error', cv=5, n_jobs = -1)
    print(np.mean(scores))


    """
    with measure_time('Training'):
        print('Training...')
        model.fit(X_ls, y_ls)


    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Build the prediction matrix
    X_ts = create_learning_matrices(rating_matrix, test_user_movie_pairs)

    # Predict
    print("Predict..")
    y_pred = model.predict(X_ts)
    "print(mean_squared_error(y_ts, y_pred))"

    # Making the submission file
    file_name =  os.path.basename(sys.argv[0]).split(".")[0]
    fname = make_submission(y_pred, test_user_movie_pairs, file_name)
    print('Submission file "{}" successfully written'.format(fname))
    """