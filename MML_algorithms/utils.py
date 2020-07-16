import numpy as np

def flip_svd(U, V, U_based=True):
    if U_based:
        max_values_idx_col = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_values_idx_col, range(U.shape[1])])
        U *= signs
        V *= signs[:, np.newaxis]
    else:
        max_values_idx_row = np.argmax(np.abs(V), axis=1)
        signs = np.sign(U[range(V.shape[0]), max_values_idx_row])
        U *= signs
        V *= signs[:, np.newaxis]

def check_data(data):
    pass


def euclidean_distance(vect1, vetc2):
    '''

    :param vect1:
    :param vetc2:
    :return:
    '''
    return np.sqrt(np.sum((vect1 - vetc2) ** 2, axis=-1))