from __future__ import absolute_import
import numpy as np
RTOL = 1E-4
ATOL = 1E-7
TOL = 1E-6
def random_full_rank_sym_mat(dim):
    shape = [dim]*2
    
    A = np.random.rand(*shape)
    A =  0.5*(A + A.T)
    A = A + dim*np.eye(dim)
    print("Matrix rank of A: {}".format(np.linalg.matrix_rank(A)))
    print("Is A symmetric??? {}".format(np.allclose(A,A.T)))
    print("A")
    print(A)
    print("Max difference A-A.T = {}".format(np.abs(A.T-A).max()))
    
    return A


def normalize(v):
    norm = np.linalg.norm(v)
    if norm<1E-6: 
       return v
    return v / norm

def normalize_matrix(A):
    return np.vstack([normalize(col) for col in A.T]).T

def is_zero_matrix(A):
    return np.allclose(A, np.zeros(A.shape), rtol=RTOL, atol=ATOL)

def is_orthogonal_matrix(A):
    return np.allclose(A.T @ A, np.eye(A.shape[1]), rtol=RTOL, atol=ATOL)
