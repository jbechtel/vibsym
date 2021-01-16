import attr
import typing as T
import numpy as np
from .SymRep import SymRep
from .repgen import trans_rota_basis_2D
import logging

logger = logging.getLogger(__name__)


@attr.s
class Molecule:
    coords: np.ndarray = attr.ib()
    rep: SymRep = attr.ib()

    def find_normal_modes(self) -> T.Tuple[np.ndarray, T.Sequence]:
        """ Block diagonalizes and reorients sym-adapted basis of self.rep

        :return: Symmetry-adapted basis vectors as columns in matrix Q and
            a list of the size of the subsapce dimensions
        """
        logger.info(" is REP a group: {}".format(self.rep.is_group()))
        trans_rota_Q = trans_rota_basis_2D(self.coords)
        logger.info('trQ  = {}'.format(trans_rota_Q))
        logger.info(' is big rep a group? {}'.format(self.rep.is_group()))
        Q, dims = self.rep.block_diagonalize(trans_rota_Q)
        logger.info(f'Q:\n{Q}')
        logger.info(f'dims:\n{dims}')
        logger.info("dims : {}".format(dims))
        subspaces, Q = self.rep.find_high_symmetry_directions(Q, dims)
        logger.info("Q:\n{}".format(Q))
        logger.info("Q.T @ Q :\n{}".format(Q.T @ Q))
        return Q, dims
