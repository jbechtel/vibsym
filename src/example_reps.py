import attr
import numpy as np
from enum import IntEnum, auto
from .repgen import generate_2D_cartesian_representation
from .SymRep import SymRep


class ExampleMoleculeType(IntEnum):
    EQUILATERAL_TRIANGLE = auto()
    HEXAGON = auto()
    SQUARE = auto()


@attr.s
class Molecule:
    coords: np.array
    rep: SymRep


_EXAMPLE_MOLECULES = {
    ExampleMoleculeType.EQUILATERAL_TRIANGLE: Molecule(
        coords=[[0., 1.],
                [np.cos(210. * np.pi / 180.), np.sin(210 * np.pi / 180)],
                [np.cos(-30. * np.pi / 180), np.sin(-30. * np.pi / 180.)]],
        rep=generate_2D_cartesian_representation(
            [[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1], [2, 1, 0], [1, 0, 2]],
            [0, 120, -120],
            [90, 210, -30])
    ),
    ExampleMoleculeType.SQUARE: Molecule(
        coords=[[1., 1.], [-1., 1.], [-1., -1.], [1., -1]],
        rep=generate_2D_cartesian_representation(
            [[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0],
             [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3], [3, 2, 1, 0]],
            [0, 90, 180, 270],
            [45, 90, 135, 180])
    ),
    ExampleMoleculeType.HEXAGON: Molecule(
        coords=[[np.sqrt(3.) / 2., 0.5], [0., 1.], [-np.sqrt(3.) / 2., 0.5],
                [-np.sqrt(3.) / 2., -0.5], [0., -1.], [np.sqrt(3.) / 2., -0.5]],
        rep=generate_2D_cartesian_representation(
            [[0, 1, 2, 3, 4, 5], [5, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 0],
             [4, 5, 0, 1, 2, 3], [2, 3, 4, 5, 0, 1], [3, 4, 5, 0, 1, 2],
             [2, 1, 0, 5, 4, 3], [3, 2, 1, 0, 5, 4], [4, 3, 2, 1, 0, 5],
             [5, 4, 3, 2, 1, 0], [0, 5, 4, 3, 2, 1], [1, 0, 5, 4, 3, 2]],
            [0, 60, -60, 120, -120, 180],
            [90, 120, 150, 180, 210, 240])
    ),
}


def get_example_molecule(molecule_type: ExampleMoleculeType) -> Molecule:
    return _EXAMPLE_MOLECULES[molecule_type]
