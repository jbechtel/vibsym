# vibsym

Vibrational symmetry methods for 2D molecules. 

![](https://github.com/jbechtel/vibsym/blob/master/runs/input.gif "Square Molecule Normal Modes")

This purpose of this project is to build a symmetry representation class `SymRep` that can find the normal modes of vibration for a 2D molecule. Finding the normal modes of a molecule is equivalent to block diagonalizing the full cartesian symmetry representation (i.e. the direct sum of the permutation and vector representations of the group). 

## Setup
Use `bash ./tools/setup_pyenv.sh` to install the correct version of python and to get a virtual env up and running.

## Usage

The `SymRep` class contains the most important algorithms and can be imported via 
```
from vibsym.SymRep import SymRep
```

To find a symmetry adapted basis of normal modes one simply needs to run `SymRep.block_diagonalize()` on an initialized `SymRep`. This algorithm is based on appendix A of [this paper](https://www.sciencedirect.com/science/article/pii/S0022509616309309).

The symmetry adapted basis can be reoriented within each subspace to align with high symmetry directions. This is implemented in the `SymRep.find_high_symmetry_direction()` method and follows an algorithm based on appendix B of the paper linked above. 

These methods are wrapped wrapped by the method `SymRep.find_normal_modes()`

Make a plot of an example molecule's normal modes using `vibsym.plotter.plot_normal_modes`



