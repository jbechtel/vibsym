import numpy as np
from .SymRep import SymRep
import itertools
import logging

logger = logging.getLogger(__name__)


def generate_2D_triangle_cartesian_representation():
    """ this generates a set of 6x6 matrices for a 2D triangle. """
    tri_rep = set()
    # symmetry operations include
    # identity, 120 degree rotation, -120 degree rotation, 3 mirrors
    permutations = [ [ 0,1,2], [2,0,1], [1,2,0], [0,2,1], [2,1,0], [1,0,2] ]
    rotations = [ 0, 120, -120]
    reflections = [ 90, 210, -30]
    rota_mats = [generate_2D_rotation(angle) for angle in rotations]
    refl_mats = [generate_2D_reflection(angle) for angle in reflections]
    cart_rep= rota_mats + refl_mats
    #cart_rep= rota_mats
    cartSymRep = SymRep(cart_rep)
    logger.debug('is cart rep a group? {}'.format(cartSymRep.is_group()))
    for i,op in enumerate(cart_rep):
        logger.debug("cart op #{}:\n{}".format(i,op))
    logger.debug(' mult table for cart rep\n{}'.format(cartSymRep.multiplication_table()))
    perm_rep = [generate_permutation_matrix(perm) for perm in permutations]
    permSymRep = SymRep(perm_rep)
    logger.debug('is perm rep a group? {}'.format(permSymRep.is_group()))
    logger.debug(' mult table for perm rep\n{}'.format(permSymRep.multiplication_table()))
    for i,op in enumerate(perm_rep):
        logger.debug("perm op #{}:\n{}".format(i,op))
    tri_rep = permuted_direct_sum(cart_rep, perm_rep)
    triSymRep = SymRep(tri_rep)
    logger.debug(' mult table for tri rep\n{}'.format(triSymRep.multiplication_table()))

    return tri_rep

def generate_2D_square_cartesian_representation():
    """ this generates a set of 8x8 matrices for a 2D triangle. """
    # symmetry operations include
    # identity, 120 degree rotation, -120 degree rotation, 3 mirrors
    permutations = [ [ 0,1,2,3], [3,0,1,2], [2,3,0,1], [1,2,3,0], [0,3,2,1], [1,0,3,2],[2,1,0,3],[3,2,1,0] ]
    rotations = [ 0, 90, 180, 270]
    reflections = [ 45, 90, 135, 180]
    rota_mats = [generate_2D_rotation(angle) for angle in rotations]
    refl_mats = [generate_2D_reflection(angle) for angle in reflections]
    cart_rep= rota_mats + refl_mats
    #cart_rep= rota_mats
    cartSymRep = SymRep(cart_rep)
    logger.debug('is cart rep a group? {}'.format(cartSymRep.is_group()))
    for i,op in enumerate(cart_rep):
        logger.debug("cart op #{}:\n{}".format(i,op))
    logger.debug(' mult table for cart rep\n{}'.format(cartSymRep.multiplication_table()))
    perm_rep = [generate_permutation_matrix(perm) for perm in permutations]
    permSymRep = SymRep(perm_rep)
    logger.debug('is perm rep a group? {}'.format(permSymRep.is_group()))
    logger.debug(' mult table for perm rep\n{}'.format(permSymRep.multiplication_table()))
    for i,op in enumerate(perm_rep):
        logger.debug("perm op #{}:\n{}".format(i,op))
    tri_rep = permuted_direct_sum(cart_rep, perm_rep)
    triSymRep = SymRep(tri_rep)
    logger.debug(' mult table for tri rep\n{}'.format(triSymRep.multiplication_table()))

    return tri_rep

def generate_2D_cartesian_representation(permutations,rotations,reflections):
    """ this generates a set of 8x8 matrices for a 2D triangle. """
    # symmetry operations include
    # identity, 120 degree rotation, -120 degree rotation, 3 mirrors
    #permutations = [ [ 0,1,2,3], [3,0,1,2], [2,3,0,1], [1,2,3,0], [0,3,2,1], [1,0,3,2],[2,1,0,3],[3,2,1,0] ]
    #rotations = [ 0, 90, 180, 270]
    #reflections = [ 45, 90, 135, 180]
    rota_mats = [generate_2D_rotation(angle) for angle in rotations]
    refl_mats = [generate_2D_reflection(angle) for angle in reflections]
    cart_rep= rota_mats + refl_mats
    #cart_rep= rota_mats
    cartSymRep = SymRep(cart_rep)
    logger.debug('is cart rep a group? {}'.format(cartSymRep.is_group()))
    for i,op in enumerate(cart_rep):
        logger.debug("cart op #{}:\n{}".format(i,op))
    logger.debug(' mult table for cart rep\n{}'.format(cartSymRep.multiplication_table()))
    perm_rep = [generate_permutation_matrix(perm) for perm in permutations]
    permSymRep = SymRep(perm_rep)
    logger.debug('is perm rep a group? {}'.format(permSymRep.is_group()))
    logger.debug(' mult table for perm rep\n{}'.format(permSymRep.multiplication_table()))
    for i,op in enumerate(perm_rep):
        logger.debug("perm op #{}:\n{}".format(i,op))
    tri_rep = permuted_direct_sum(cart_rep, perm_rep)
    triSymRep = SymRep(tri_rep)
    logger.debug(' mult table for tri rep\n{}'.format(triSymRep.multiplication_table()))

    return tri_rep


def generate_2D_rotation(angle):
    """ Given an angle generate a 2D rotation matrix"""
    angle_rad = angle/180.*np.pi
    return np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]])

def generate_2D_reflection(angle):
    """ Given an angle generate a 2D rotation matrix"""
    angle_rad = angle/180.*np.pi
    return np.array([[np.cos(2.*angle_rad),np.sin(2.*angle_rad)],[np.sin(2.*angle_rad),-np.cos(2.*angle_rad)]])

def generate_permutation_matrix(permutation):
    size = len(permutation)
    ordered_perm = list(range(size))
    perm_matrix = np.zeros((size,size))
    for i,j in zip(ordered_perm,permutation):
        perm_matrix[i,j]=1.

    return perm_matrix

def generate_all_permutation_matrices(n):
    # there are n! permutations
    # here is a list of them all
    all_perms = list(itertools.permutations(np.arange(n)))
    all_perm_mats = [ generate_permutation_matrix(perm) for perm in all_perms ]
    return all_perm_mats

def permuted_direct_sum(cart_rep, perm_rep):
    full_rep = [np.kron(perm, cart) for perm,cart in zip( perm_rep,cart_rep)]
    return full_rep

def trans_rota_basis_2D(p):
    # define points
    logger.debug(f'points: \n{p}')
    #p = [[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1]]
    # unit translations in x and y
    trans = [ [ 1.0, 0],[0.,1.]]
    # small rotation about z
    #rota = generate_2D_rotation(.00001) # this is wrong
    # this is a counter clockwise infinitesimal rotation
    rota = np.array([ [0,-1.],[1.,0]])
    # now create a 2N X 3 matrix where the first two columns represent unit translations, and the last
    # represents a rotation
    # the orthogonal complement to this subspace defines the subspace for internal vibrations
    Qt = []
    for t in trans:
        Qt.append(t*len(p))

    # now we want to apply the rotation to each of the coordinate vectors, and find the vectors that
    # connect the original coordinate to the rotated coordinate. This gives the
    # displacement field associated with a rotation
    rotated_p = [ np.dot(np.array(orig),rota.T) for orig in p]
    Rt=[]
    for rp,op in zip(rotated_p,p):
        #Rt.extend(rp - np.array(op))
        Rt.extend(rp)
    logger.debug('Rt=\n{}'.format(Rt))
    logger.debug(f'Qt before Rt: {Qt}')
    #Rt = flatten(Rt)
    Qt.append(Rt)
    # Qt.append([direction for atom in p for direction in atom])
    Q = np.array(Qt).T
    logger.debug('Q=\n{}'.format(Q))
    return Q


