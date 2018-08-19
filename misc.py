from __future__ import absolute_import
import numpy as np

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
