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
from MFclass import MF

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

    #pour Francois
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
    directory = "reconstructed"
    if not os.path.exists(directory):
        os.makedirs(directory)

    nb_iterations = [1000]
    for i in nb_iterations:
        print(i)
        print("="*10)
        rating_matrix = build_rating_matrix(user_movie_rating_triplets)
        model = MF(rating_matrix.toarray(), 30, 0.001, 0.01, i)
        model.train()

        reconstructed = model.full_matrix()
        np.savetxt('reconstructed/reconstructed_mat_30_0002_001_{}.txt'.format(i), reconstructed, fmt='%f')
