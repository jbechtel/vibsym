# vibsym

Vibrational symmetry methods for 2D molecules. 

This purpose of this project is to build a symmetry representation class `SymRep` that can find the normal modes of vibration for a 2D molecule. Finding the normal modes of a molecule is equivalent to block diagonalizing the full cartesian symmetry representation (i.e. the direct sum of the permuation and vector representations of the group). In order to find the symetry adapted basis that block diagonalizes the full Cartesian Sym Rep, several steps must be considered. 

First, the symmetry operations of the molecule must be enumerated. The `SymRep.generate_2D_molecule_point_group(coordinates)` routine will find all of the 2D symmetry operations given a set of coordinates. The algorithm is a brute force approach that proceeds as follows:
```
    def generate_2D_molecule_point_group(self,coordinates):
        """ calculate isometries of 2D molecule 

            This assumes that:
                1. Atoms are all same species!
                2. Only working in 2D
                3. Molecule is centered on origin

        The algorithm is as follows:
            - cycle through all pairs of atoms to construct a matrix R = [ atom1 atom2 ]
                where atom1, atom2 etc are column vectors of atomic coordinates 
            - determine the rank of R, if it is < 2, then the atoms are collinear.
                if they are exactly on the opposite side of the origin,
                i.e., atom1 = -atom2, then there are two operations that may 
                leave them invariant: 180 rotation, or a reflection about the 
                line connecting them.
                Hence in this case both of these ops will be appended, 
                else we move on.
            - we will be working with the basic relation: S*R = R*P
                where S is an orthogonal matrix (rotation or reflection)
                and P is an orthogonal permuation matrix.
                This says that reordering atoms is equivalent to some spatial
                isometry which is the fundamental essence of a symmetry operation
            - We enumerate the possible permutations, other than identity,
                which in 2D is just P = np.array([[0.,1.],[1.,0.]])
            - Then calculate S = R P inv(R) 
                and add to the list of possible isometries.
            - Finally, we apply the list of possible isometries and 
                test if it is equivalent to a permutation. 
                So we also need a list of all possible permutation matrices.
                Then test whether S R = R P
        """
```
Given the permutation representation and the vector represetntation the full Cartesian symmetry representation is built using a direct sum using helper functions found in the `repgen` module i.e. `repgen.permuted_direct_sum(perm_rep,cart_rep)`.

Next, an adapted basis is found using the `SymRep.block_diagonalize()` routine which is detailed in appendix A of [this paper](https://www.sciencedirect.com/science/article/pii/S0022509616309309) from our group.

Lastly, the symmetry adapted basis can be reoriented within each subspace to align with high symmetry directions. This is implemented in the `SymRep.find_high_symmetry_direction()` method and follows an algorithm based on appendix B of the paper linked above. 

Finally, the normal modes can be animated using one of the plotting scripts in `scripts`. 
