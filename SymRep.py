from __future__ import absolute_import
import numpy as np
from . import repgen 
from . import misc 
import itertools

class SymRep(object):
    def __init__(self,full_rep=None,coordinates=None):
        """ full rep is a list of numpy arrays that form a group """
        if full_rep is not None:
            self.full_rep = full_rep
            self.dim = full_rep[0].shape[0]
            self.tol=1E-8
        elif coordinates is not None:
            self.cart_rep,self.perm_rep = self.generate_2D_molecule_point_group(coordinates)
            self.pretty_print_cart()
            self.pretty_print_perm()
            self.full_rep = repgen.permuted_direct_sum(self.cart_rep,self.perm_rep)
            self.dim = self.full_rep[0].shape[0]
            self.tol=1E-8
        else: full_rep = None



    def block_diagonalize(self,trans_rota_subspace=None):
        assert  self.full_rep is not None, "Error: trying to block diagonalize a representation that hasn't been intialized"
        np.set_printoptions(precision=3,suppress=True)
        
        # get orthogonal complement of the subspace spanned by translations
        # and rotations. As usual find the SVD, A = U D V
        # where U is orthogonal, D are singular values, V spans range
        if trans_rota_subspace is not None:
            u,s,vh = np.linalg.svd(trans_rota_subspace)
            W = u[:,trans_rota_subspace.shape[1]:]
            print(' DOT between W.T @ trans_rota_subspace \n{}'.format(W.T @ trans_rota_subspace))
            print('W shape trans_rota = {}'.format(W.shape))
            tA = W @ misc.random_full_rank_sym_mat(W.shape[1]) @ W.T 
            print("Matrix rank of tA: {}".format(np.linalg.matrix_rank(tA)))
            print("tA")
            print(tA)
            num_dofs = W.shape[1]
            #num_dofs = W.shape[0] - W.shape[1]
        else:
            tA = misc.random_full_rank_sym_mat(self.dim)
            num_dofs = self.dim


        Qt=[]
        dims = []
        count = 1
        while len(Qt) < num_dofs:
            
            A = self.reynolds(tA/tA.shape[0])  
            print(' is A symmetric after Reynolds?')
            print("Matrix rank of A: {}".format(np.linalg.matrix_rank(A)))
            print("Is A symmetric??? {}".format(np.allclose(A,A.T)))
            print("A")
            print(A)
            print("Max difference A-A.T = {}".format(np.abs(A.T-A).max()))
            print('is A invariant? {}'.format(self.is_invariant(A)))
            D,q = np.linalg.eigh(A) 
            print('matrix_rank of q {}'.format(np.linalg.matrix_rank(q)))
            u, indices = np.unique(np.around(D,6), return_inverse=True)
            u = u[::-1]
            indices = indices.max() - indices
            print('u')
            print(u)
            print('indices')
            print(indices)
            for i,val in enumerate(u):
                if val==0:
                    continue
                else:
                    print('val {}'.format(val))
                    print('np.where(D==val) {}'.format(np.where(indices==i)[0]))
                    print('q')
                    print(q)
                    print('q.T @ q\n{}'.format(q.T @ q))
                    print('is q orthogonal? {}'.format(np.allclose(q.T @ q, np.eye(q.shape[1]),atol=self.tol)))
                    V = q[:,np.where(indices==i)[0]]
                    print('V')
                    print(V)
                    print('is V orthogonal? {}'.format(np.allclose(V.T @ V, np.eye(V.shape[1]))))
                    temp_rep = SymRep([ V.T @ op @ V for op in self.full_rep ])
                    temp_rep.pretty_print_full()
                    if not temp_rep.is_reducible():
                        for row in V.T:
                            Qt.append(row)
                        dims.extend([count]*len(np.where(indices==i)[0]))
                        count += 1
            Q = np.array(Qt).T
            print('Q \n{}'.format(Q))
            print('matrix rank Q {}'.format(np.linalg.matrix_rank(Q)))
            #exit()
            if (np.linalg.matrix_rank(Q) < num_dofs) and  (np.linalg.matrix_rank(Q) > 0):
                u,s,vh = np.linalg.svd(Q)
                W = u[:,Q.shape[1]:]
                tA = W @ misc.random_full_rank_sym_mat(W.shape[1]) @ W.T 

        return np.array(Qt).T,dims




    

    def is_reducible(self):
        assert  self.full_rep is not None, "Error: trying to check reducibility on a representation that hasn't been intialized"
        group_size = len(self.full_rep)
        shur = 0
        for op in self.full_rep:
            for d in np.diagonal(op):
                shur+=d**2

        print(' shur: {}\t group_size: {}'.format(shur,group_size))
        print(' shur==group_size? {}'.format(abs(group_size-shur)<self.tol))
        return abs(group_size-shur)>self.tol



    def is_invariant(self,A):
        assert  self.full_rep is not None, "Error: trying to check invariance on a representation that hasn't been intialized"
        is_invariant = True
        for i,op in enumerate(self.full_rep):
            if not np.allclose(op.T @ A @ op,A,atol=self.tol):
                is_invariant = False
                print(' A is not invariant to op #{}\n{}'.format(i,op))
        return is_invariant




    def reynolds(self,A):
        assert  self.full_rep is not None, "Error: trying to apply Reynolds operator to representation that hasn't been intialized"
        tA = np.zeros(A.shape)
        rey = [ op.T @ A @ op for op in self.full_rep ]
        tA = sum(rey)
        #for op in self.full_rep:
        #    tA += op.T @ A @ op
        tA /= float(len(self.full_rep))
        print(' Reynold\'s on A :\n{}'.format(tA))
        return tA

    def is_group(self):
        assert  self.full_rep is not None, "Error: trying to check group closure on a representation that hasn't been intialized"
        I = np.eye(self.dim)
        I_in_group = False
        op_orthogonal = True
        for i,op in enumerate(self.full_rep): 
            if not np.allclose(op.T @ op, I,atol=self.tol):
                print('non orthogonal op is num {} looks like\n {}'.format(i,op))
                op_orthogonal = False
                return op_orthogonal
            if np.allclose(op,I,atol=self.tol):
                I_in_group = True
            for sop in self.full_rep:
                op_in_group = False 
                for test in self.full_rep:
                    if np.allclose(op@sop,test,atol=self.tol):
                        op_in_group = True
                if op_in_group==False:
                    return False
        return I_in_group 

    def multiplication_table(self):
        assert  self.full_rep is not None, "Error: trying to check multiplciation table on a representation that hasn't been intialized"
        mtable = np.zeros([len(self.full_rep)]*2)
        for i,op in enumerate(self.full_rep): 
            for j,sop in enumerate(self.full_rep): 
                for num,fop in enumerate(self.full_rep):
                    if np.allclose(op@sop,fop,atol=self.tol):
                        mtable[i,j] = num
        print(" mult table is orthogonal? Ans: {}".format(np.allclose(np.eye(len(self.full_rep)),mtable.T @ mtable,atol=self.tol)))
        return mtable

    def pretty_print_full(self):
        assert  self.full_rep is not None, "Error: trying to pretty print a representation that hasn't been intialized"
        np.set_printoptions(precision=3,suppress=True)
        for i,op in enumerate(self.full_rep):
            print(' op #{} \n{}'.format(i,op))
    def pretty_print_cart(self):
        assert  self.cart_rep is not None, "Error: trying to pretty print a representation that hasn't been intialized"
        np.set_printoptions(precision=3,suppress=True)
        for i,op in enumerate(self.cart_rep):
            print(' op #{} \n{}'.format(i,op))
    def pretty_print_perm(self):
        assert  self.perm_rep is not None, "Error: trying to pretty print a representation that hasn't been intialized"
        np.set_printoptions(precision=3,suppress=True)
        for i,op in enumerate(self.perm_rep):
            print(' op #{} \n{}'.format(i,op))

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
        n = len(coordinates)
        R = np.array(coordinates).T
        # possible isometries
        testS = []
        testS.append(np.eye(2))
        swap = np.array([[0.,1.],[1.,0.]])
        # now we want to enumerate all pairs
        pair_indices = list(itertools.permutations(range(n), 2))
        #tpair_indices = list(itertools.combinations(range(n), 2))
        for inds in pair_indices:
            for tinds in pair_indices:
                pairR = R[:,inds]
                tpairR = R[:,tinds]
                if np.linalg.matrix_rank(pairR) != np.linalg.matrix_rank(tpairR):
                    # a sym op wont change a pair to collinear,
                    # it must mantain local topology
                    continue
                elif (np.allclose(pairR[:,0],-pairR[:,1])) and (np.linalg.matrix_rank(pairR)<2):
                    rota = repgen.generate_2D_rotation(180.)
                    refl = repgen.generate_2D_reflection(np.arctan(pairR[0,1]/pairR[0,0])/np.pi*180.)
                    #print("Adding 180 rotation\n{}\n and reflection\n{}\n".format(rota,refl))
                    testS.append(rota)
                    testS.append(refl)
                elif np.linalg.matrix_rank(pairR)<2:
                    continue
                else:
                    #print("attempting to map inds:{} to inds:{}".format(inds,tinds))
                    #tmpS = pairR @ swap @ np.linalg.inv(pairR)
                    tmpS = tpairR @ np.linalg.inv(pairR)
                    if np.allclose(tmpS @ tmpS.T,np.eye(2),1E-6):
                        #print("Found orthogonal transformation matrix:\n{}".format(tmpS))
                        testS.append(tmpS)
                    else: 
                        continue

        # all permutations 
        testP = repgen.generate_all_permutation_matrices(n)

        mol_pg = []
        mol_perms = []
        for i,op in enumerate(testS):
            #print("op #{}\n{}".format(i,op))
            for perm in testP:
                #print("perm\n{}".format(perm))
                #print("R\n{}".format(R))
                in_group = False
                for top in mol_pg:
                    if np.allclose(op,top):
                        in_group = True
                if not in_group:
                    if np.allclose(op @ R, R @ perm):
                        mol_pg.append(op)
                        mol_perms.append(perm)
        return mol_pg,mol_perms



         


