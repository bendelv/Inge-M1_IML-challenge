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
from scipy.sparse.linalg import svds

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

if __name__ == '__main__':
    prefix = 'data/'

    #pour Frnacois
    """
        mean_users = np.zeros((rating_matrix.shape[0], 1))
        for i in np.arange(1, rating_matrix.shape[0]):
            mean_users[i] = np.mean(rating_matrix[i].data)
    """

    # ------------------------------- Learning ------------------------------- #
    training_user_movie_pairs = load_from_csv(os.path.join(prefix,
                                                           'data_train.csv'))
    training_labels = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    user_movie_rating_triplets = np.hstack((training_user_movie_pairs,
                                            training_labels.reshape((-1, 1))))

    rating_matrix = build_rating_matrix(user_movie_rating_triplets)


    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))
    y_pred = all_prediction[test_user_movie_pairs]

    file_name =  os.path.basename(sys.argv[0]).split(".")[0]
    fname = make_submission(y_pred, test_user_movie_pairs, file_name)
    print('Submission file "{}" successfully written'.format(fname))
