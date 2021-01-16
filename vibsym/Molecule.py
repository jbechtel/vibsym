import attr
import typing as T
import numpy as np
from .SymRep import SymRep
from .repgen import trans_rota_basis_2D


@attr.s
class Molecule:
    coords: np.array
    rep: SymRep

    def find_normal_modes(self) -> T.Tuple[np.ndarray,T.Sequence]:
        """ Block diagonalizes and reorients sym-adapted basis of self.rep

        :return: Symmetry-adapted basis vectors as columns in matrix Q and
            a list of the size of the subsapce dimensions
        """
        print(" is REP a group: {}".format(self.rep.is_group()))
        trans_rota_Q = trans_rota_basis_2D(self.coords)
        print('trQ  = {}'.format(trans_rota_Q))
        print(' is big rep a group? {}'.format(self.rep.is_group()))
        Q, dims = self.rep.block_diagonalize(trans_rota_Q)
        print(f'Q:\n{Q}')
        print(f'dims:\n{dims}')
        print("dims : {}".format(dims))
        subspaces, Q = self.rep.find_high_symmetry_directions(Q, dims)
        print("Q:\n{}".format(Q))
        print("Q.T @ Q :\n{}".format(Q.T @ Q))
        return Q, dims
