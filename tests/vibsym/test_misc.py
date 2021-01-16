import numpy as np
from vibsym.misc import random_full_rank_sym_mat, normalize, normalize_matrix, \
    is_zero_matrix, is_orthogonal_matrix

def test_random_full_rank_sym_mat():
    A = random_full_rank_sym_mat(10)
    assert np.allclose(A, A.T)
    assert np.linalg.matrix_rank(A) == len(A)


