from __future__ import absolute_import
import numpy as np
from . import repgen 
from . import misc 
import itertools

class SymRep(object):
    """ 
    Symmetry Reprentation Class has symmetry and linear algebra algorithms
        that can be used to investigate the symmetry properties of 
        2D objects. 


    """
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
        """        # what is source of this algorithm?
        probably defined in jct's paper  The exploration of nonlinear elasticity and its efficient parameterization for crystalline materials
        idea of complement to disp/rota in Finite-temperature properties of strongly anharmonic and mechanically unstable crystal phases from first principles
        """
        assert  self.full_rep is not None, "Error: trying to block diagonalize a representation that hasn't been intialized"
        np.set_printoptions(precision=3,suppress=True)
        
        # get orthogonal complement of the subspace spanned by translations
        # and rotations. As usual find the SVD, A = U D V
        # where U is orthogonal, D are singular values, V spans range
        # the complement of A is given by U[:,A.shape[1]:] i.e. the columns that 
        # are not spanned by A. 
        if trans_rota_subspace is not None:
            u,s,vh = np.linalg.svd(trans_rota_subspace)
            # get null space of U
            W = u[:,trans_rota_subspace.shape[1]:]
            # this should be orthogonal
            print(' DOT between W.T @ trans_rota_subspace \n{}'.format(W.T @ trans_rota_subspace))
            assert misc.is_zero_matrix(W.T @ trans_rota_subspace), 'complement not orthogonal to trans_rota_subspace'
            print('W shape trans_rota = {}'.format(W.shape))
            # this is a coordinate transformation that makes a random full rank matrix 
            # in the space of purely deformational displacements .
            tA = W @ misc.random_full_rank_sym_mat(W.shape[1]) @ W.T
            tA_rank = np.linalg.matrix_rank(tA)
            print("Matrix rank of tA: {}".format(tA_rank))
            assert tA_rank == W.shape[1]
            print("tA")
            print(tA)
            num_dofs = W.shape[1]
        else:
            tA = misc.random_full_rank_sym_mat(self.dim)
            num_dofs = self.dim


        # Qt holds transposed column vectors of Q(e->E)
        Qt=[]
        dims = []
        count = 1
        print(f'num_dofs: \n{num_dofs}')
        while len(Qt) < num_dofs:
            print(f'len(Qt): {len(Qt)}')
            
            # why  divide by shape?
            A = self.reynolds(tA)  
            print(' is A symmetric after Reynolds?')
            print("Matrix rank of A: {}".format(np.linalg.matrix_rank(A, tol=TOL)))
            print("Is A symmetric??? {}".format(np.allclose(A,A.T)))
            print("A")
            print(A)
            print("trans_rota_subspace")
            print(trans_rota_subspace)
            print("Max difference A-A.T = {}".format(np.abs(A.T-A).max()))
            print('is A invariant? {}'.format(self.is_invariant(A)))
            print('trans_rota_Q.T @ A: \n{}'.format(trans_rota_subspace.T @ A))

            # eigenvalues, D, and eigenbasis q
            D,q = np.linalg.eigh(A) 
            print('matrix_rank of q {}'.format(np.linalg.matrix_rank(q)))
            # find the counts of unique eigenvalues
            # trying to find those grops that span a subspace 
            u, indices = np.unique(np.around(D,6), return_inverse=True)
            # reverse the order of the eigenvalues and indices to go from largest to smallest
            u = u[::-1]
            indices = indices.max() - indices
            print('u')
            print(u)
            print('indices')
            print(indices)
            for i,val in enumerate(u):
                if abs(val)<TOL:
                    continue
                else:
                    print('val {}'.format(val))
                    print('np.where(D==val) {}'.format(np.where(indices==i)[0]))
                    print('q')
                    print(q)
                    print('q.T @ q\n{}'.format(q.T @ q))
                    print('is q orthogonal? {}'.format(misc.is_zero_matrix(q.T @ q - np.eye(q.shape[1]))))
                    V = q[:,np.where(indices==i)[0]]
                    print('V')
                    print(V)
                    print('is V orthogonal? {}'.format(misc.is_zero_matrix(V.T @ V - np.eye(V.shape[1]))))
                    temp_rep = SymRep([ V.T @ op @ V for op in self.full_rep ])
                    temp_rep.pretty_print_full()
                    if not temp_rep.is_reducible():
                        print(f'len(V.T):\n{len(V.T)}')
                        print(f'V.T:\n{V.T}')
                        for row in V.T:
                            Qt.append(row)
                        print(f'count: {count}')
                        print(f'curr indices: {len(np.where(indices==i)[0])}')
                        dims.extend([count]*len(np.where(indices==i)[0]))
                        #dims.append(V.shape[1])
                        count += 1
            Q = np.array(Qt).T
            print('Q \n{}'.format(Q))
            print('matrix rank Q {}'.format(np.linalg.matrix_rank(Q)))
            # exit()
            if (np.linalg.matrix_rank(Q) < num_dofs) and  (np.linalg.matrix_rank(Q) > 0):
                u,s,vh = np.linalg.svd(Q)
                # get orthogonal complement
                W = u[:,Q.shape[1]:]
                # new random matrix that spans it
                tA = W @ misc.random_full_rank_sym_mat(W.shape[1]) @ W.T 
        assert len(dims) == num_dofs
        Q = np.array(Qt).T
        assert misc.is_zero_matrix((Q.T @ Q) - np.eye(Q.shape[1]))
        # assert (np.abs(np.linalg.det(Q))-1) < TOL
        return Q,dims

    def find_high_symmetry_directions(self,Q=None,irrep_lookup=None): # dims=irrep_indices
        """ Need to enumerate all subgroups G_i, of P.  
            Then for each representation, average the action of the subgroup
            and observe the column space of the result. If it is 1, then there is only 
            one direction invariant to G_i, and the rest of the equivalent directions can 
            be found by applying the whole group P. If the high symmetry directions 
            are mutually orthogonal and span the subspace, then they are chosen as the
            oriented basis. This will essentially just be a rotation of certain rows/columns
            of Q. If it is a 2D space such as e2-e3 strain space, then the spanning set of high 
            symm directions will not form an orthogonal set. In this case just chose as many
            high symmetry directions that are orthogonal, and get the rest from the 
            orthogonal complement.
        
        """
        # use Q to block diagonalize the full representation  
        block_rep = [ Q.T @ op @ Q for op in self.full_rep]  
        # visualize blockdiagonalized representation for debugging  
        for op in block_rep:
            print(" op in block_rep : \n{}\n".format(op))
            assert np.abs(np.linalg.det(op)) - 1 < TOL

        # get all subgroups 
        # subgroups is a list of list of indices into self.full_rep
        # that form a subgroup
        subgroups = self.find_all_subgroups() 
        print(len(subgroups))

        # this is an awkward counter that counts the entries in irrep_lookup
        # likely, another container is perferred to irrep_lookup.
        bindex = 0

        # dims is a list specifying to which 
        # irrep each normal mode belongs. 
        # i.e. [1,2,2,3] means there is 
        # a 1-D irrep at index 0, then a 2D irrep 
        # spanned by normal modes 1 & 2, then a 1D
        # irrep spanned by normal mode 3.
        # print("dims: {}".format(dims))
        print("irrep_lookup: {}".format(irrep_lookup))
        # get the unique irrep labels 
        # udims = np.unique(dims)
        irrep_labels = np.unique(irrep_lookup)
        # subspaces 
        subspaces = []
        newQ = []
        tmplistQ = []

        # iterate through irreducible subspaces 
        for irrep_label in irrep_labels:
            # get the normal modes associated 
            # with the current irrep 
            irrep_inds = sorted(list(np.where(irrep_lookup==irrep_label)[0]))
            # Get normal modes 
            print("irrep_inds : {}".format(irrep_inds))
            curr_Qs = Q[:,irrep_inds]
            # if the irrep is 1D, then we don't need to 
            # find a high symmetry direction.
            # subgroup_invariant_directions

            if len(irrep_inds)==1:
                for row in curr_Qs.T:
                    newQ.append(row)
                print('irrep_inds==1')
                print("newQ : \n{}".format(newQ))
                bindex+=1
                continue
            # subgroup is a list of indices into self.full_rep 
            # or block_rep for that matter
            list_of_orbits = []
            for subgroup in subgroups:
                print(f'irrep_inds=={len(irrep_inds)}')
                orbits = []
                sublistQ = []
                # get this subspace's rep. 
                curr_subspace_rep =[ op[bindex:bindex+len(irrep_inds),bindex:bindex+len(irrep_inds)] for op in block_rep]
                curr_subgroup_rep =[ curr_subspace_rep[i] for i in subgroup]
                # reynolds
                np.set_printoptions(precision=8)
                print(f'curr_subgroup_rep: \n{curr_subgroup_rep}')
                averaged = np.sum(curr_subgroup_rep, axis=0) / len(curr_subgroup_rep)
                print(f'averaged: \n{averaged}')
                averaged = misc.normalize_matrix(averaged)
                print(f'normalized averaged: \n{averaged}')
                gram = averaged.T @ averaged
                print(f'gram:\n{gram}')
                # see if there were any non zero directions
                rank = np.linalg.matrix_rank(averaged, tol=1E-6)
                print(f'rank: {rank}')
                if (rank==0) or (rank==averaged.shape[0]):
                    print('no high sym directions projected out for this subgroup')
                    continue
                q,r = np.linalg.qr(averaged)
                print(f'q: \n{q}')
                print(f'r: \n{r}')
                # Row indices of non-zero elements of r give linear independent cols of q. 
                # row_indices, col_indices = np.where(np.abs(r)>1E-6)

                sublistQ = unique_vectors_from_list([vec for vec in averaged.T])
                sublistQ = [misc.normalize(vec) for vec in sublistQ if not np.linalg.norm(vec)<self.tol]

                print(f'sublistQ: {sublistQ}')
                # sublistQ holds all non-zero directions for this subgroup
                assert len(sublistQ)==rank, 'rank not equal to number of collected vecs'
                print(sublistQ)
                # for each vec in sublistQ apply the whole group, then get the 
                # set of distinct vectors. and see if they for a mutually orthogonal set with 
                # order equal to the dimension of the irrep.
                # if we find an example then we are done. OTherwise we want to find a set of mutually
                # orthogonal vectors, and then get orthogonal complement. 
                vec = sublistQ[0]
                orbit = [rep.dot(vec) for rep in curr_subspace_rep]
                print(f'orbit:\m: {orbit}')
                print(f'orbit:\m: {len(orbit)}')
                set_orbit = unique_vectors_from_list(orbit)
                print(f'set(orbit):\m: {set_orbit}')
                print(f'set(orbit):\m: {len(set_orbit)}')
                list_of_orbits.append(set_orbit)
                # if len(set_orbit)==len(irrep_inds):
                #     print('found good basis')
                # end for over invariant directions

            # now have list of orbits
            # First check if any of them are divisble by the irrep dimension
            orbits_divisible_by_irrep_dim = [o for o in list_of_orbits if len(o) % len(irrep_inds) == 0]
            print(f'orbits_divisible_by_irrep_dim : \n{orbits_divisible_by_irrep_dim}')
            if len(orbits_divisible_by_irrep_dim)>0:
                # these must? have orthogonal vecs
                orthog_bases = [b for o in orbits_divisible_by_irrep_dim for b in get_all_right_handed_2d_bases_from_orbit(o)]
                print(f'orthog_bases : \n {orthog_bases}')
                
                # collect reoriented Qs
                reoriented_Qs = []
                for orthog_basis in orthog_bases:
                    orthog_basis = orthog_basis.T
                    Q_basis = curr_Qs @ orthog_basis
                    reoriented_Qs.append(Q_basis)
                # now count number of non-zero elements in in reoriented_Q
                # this is a sparsity score we want most sparse
                num_nonzero = [(np.abs(b) < 1E-8).sum() for b in reoriented_Qs]
                best_basis_index = np.argsort(num_nonzero)[-1]
                orthog_basis = orthog_bases[best_basis_index]
                # subspaces.append(orthog_bases[best_basis_index])
                # Q_basis = roriented_Qs[best_basis_index])
                # print("Q_basis : \n{}".format(Q_basis))
                # print("before newQ : \n{}".format(newQ))
                # for col in Q_basis.T:
                #     newQ.append(col)
                # print("after newQ : \n{}".format(newQ))
            else:
                # if no orbits are divisible by irrep dim then we have too many
                # vecs for the space. 
                # just take first?
                orbit = list_of_orbits[-1]
                reoriented_Qs = []
                for v in orbit:
                    v = v.T
                    q = curr_Qs @ v
                    reoriented_Qs.append(Q)

                best_vec_index = np.argsort([(np.abs(q)< 1E-8).sum() for q in reoriented_Qs])[-1]
                print(best_vec_index)
                print(reoriented_Qs)
                col_vec = orbit[best_vec_index][:,None]
                u,d,v = np.linalg.svd(col_vec)
                orthog_vec = u[:,-1]
                print(f'col_vec: \n{col_vec}')
                print(f'orthog_vec: \n{orthog_vec}')
                orthog_basis = np.vstack((col_vec.T,u[:,-1]))# works bc only 2d irreps




            # end for over subgroups
            # now we have all of the high sym orbits in list_of_orbits 
            # # let's make a unique set of this list
            # flat_list_of_orbits = list(itertools.chain.from_iterable(list_of_orbits))
            # unique_invariant_directions = unique_vectors_from_list(flat_list_of_orbits)
            # # now we want to count how many times each unique vector occurs in flat_list_orbits
            # # the idea is that if it shows up more times it is likely a more important direction
            # counts_of_invariant_directions = count_unique_vectors(flat_list_of_orbits, unique_invariant_directions)
            # print(f'counts_of_invariant_directions: \n {counts_of_invariant_directions}')
            # print(f'unique_invariant_directions: \n {unique_invariant_directions}')
            # print(f'flat_list_of_orbits: \n {flat_list_of_orbits}')
            # print(f'len(unique_invariant_directions): \n {len(unique_invariant_directions)}')
            # print(f'len(flat_list_of_orbits): \n {len(flat_list_of_orbits)}')

            # # given few options of high multiplicity find if an orthogonal basis exists
            # potential_bases = []
            # potential_mults = []
            # zipped_vecs_mults = zip(itertools.combinations(unique_invariant_directions, 
            #                                                len(irrep_inds)), 
            #                         itertools.combinations(counts_of_invariant_directions, 
            #                                                len(irrep_inds))) 

            # for combination, mults in zipped_vecs_mults:
            #     # construct potential basis
            #     B = np.array(combination)
            #     is_orthog = misc.is_zero_matrix((B @ B.T) - np.eye(*B.shape))
            #     if is_orthog:
            #         potential_bases.append(combination)
            #         potential_mults.append(mults)
             
            # orthog_basis = None
            # bases_that_include_max_direction = [basis for basis,mult in zip(potential_bases,potential_mults) if max(counts_of_invariant_directions) in mult]
            # if len(bases_that_include_max_direction) == 0:
            #     print('no direction includes max direction')
            #     print('constructing basis from orthogonal complement')
            #     max_direction_index = np.argsort(np.array(counts_of_invariant_directions))[-1]
            #     col_vec = unique_invariant_directions[max_direction_index][:,None]
            #     u,d,v = np.linalg.svd(col_vec)
            #     orthog_vec = u[:,-1]
            #     print(f'col_vec: \n{col_vec}')
            #     print(f'orthog_vec: \n{orthog_vec}')
            #     orthog_basis = np.vstack((col_vec.T,u[:,-1]))# works bc only 2d irreps
            # elif len(potential_bases) > 1:
            #     mults = np.array([sum(m) for m in potential_mults])
            #     inds = mults.argsort()
            #     print(f'inds: \n{inds}')
            #     print(f'mults: \n{mults}')


            #     print(f'many potentials: {len(potential_bases)}')
            #     orthog_basis = np.array(potential_bases[inds[-1]])

            #     # for combination,sum_mults in zip(potential_bases,potential_mults):  
            #     #     print(combination)
            #     #     for row in combination:
            #     #         if np.allclose(np.abs(row), np.array([0,1]), atol=ATOL, rtol=RTOL):
            #     #             orthog_basis = np.array(combination)

            # elif len(potential_bases)==1:
            #     print('single potential')
            #     orthog_basis = np.array(combination)
            # else:
            #     print('somethings wrong')
            #     print(f'potential_basis: \n{potential_bases}')
            #     exit()

        # end for over irreps 
            print(f'orthog_basis: \n{orthog_basis}')




            orthog_basis = orthog_basis.T
            subspaces.append(orthog_basis)
            Q_basis = curr_Qs @ orthog_basis
            print("Q_basis : \n{}".format(Q_basis))
            print("before newQ : \n{}".format(newQ))
            for col in Q_basis.T:
                newQ.append(col)
            print("after newQ : \n{}".format(newQ))
            # for v in orbits[selected_orbit]:
            #     subspaces.append(v)

        newQ = np.array(newQ).T
        print("newQ: \n{}".format(newQ))
        print("newQ.T @ newQ\n{}".format(newQ.T @ newQ))
        return subspaces,newQ

        

        for udim in udims:
            vecs = []
            # get the normal modes associated 
            # with the current irrep 
            inds = np.where(dims==udim)[0]
            print(" inds : {}".format(inds))
            # if the irrep is 1D, then we don't need to 
            # find a high symmetry direction.
            if len(inds)==1:
                newQ.append(Q[:,inds].T.flatten())
                print("newQ : \n{}".format(newQ))
                bindex+=1
                continue
            else:
                orbits = []
                sublistQ = []
                # get this subspace's rep. 
                test_rep =[ op[bindex:bindex+len(inds),bindex:bindex+len(inds)] for op in block_rep]
                # iterate through subgroups 
                for sg in subgroups:
                    # skip identity 
                    if len(sg)==1:
                        continue
                    # we will average the sym ops  
                    # start with zerps
                    rey = np.zeros( (len(inds),len(inds)))
                    for op in sg:
                        rey += test_rep[op]
                    rey /= len(sg)
                    print("averaged over subgroup: \n{}\n".format(rey))
                    #subspaces.append(rey)
                    
                    # look at the column space of the projection  
                    # col the transposed column vector. 
                    for col in rey.T:
                        orbit=[]
                        print("col : {} ".format(col))
                        if np.allclose(col,np.zeros(col.shape)):
                            print("all zeros")
                            continue
                        else: 
                            print("not zero")
                            for op in test_rep:
                                v = op @ col.T
                                v = v / np.linalg.norm(v)
                                v_in_orbit = False
                                for t in orbit:
                                    if np.allclose(v,t):
                                        v_in_orbit = True
                                if (len(orbit)==0):
                                    print(" adding first v")
                                    orbit.append(v)
                                elif not v_in_orbit: 
                                    print("appending v to orbit" )
                                    orbit.append(v)
                        orbits.append(orbit)
                    # end subgroups

                mults = [ len(orbit) for orbit in orbits ]
                print("mults : {}".format(mults))
                sorted_mults = np.argsort(mults)[::-1]
                selected_orbit = None
                for ind in sorted_mults:
                    if len(self.full_rep) % mults[ind] == 0:
                        selected_orbit = ind
                        break
                if selected_orbit==None:
                    print(" ERROR: didn't find a high sym direction with multiplicity that divides group order ")
                    exit()
                # this gets all permutations of rep.dimension number of vectors from orbit[selected_orbit] 
                # Then use the permutations, compute the gram matrix and compare to identity
                # if identity then they form an orthogonal basis! 
                # if not, then just get an orthogonal complement.
                orthog_basis = None
                print("selected_orbit : {}".format(selected_orbit))
                print("orbits : \n{}".format(orbits))
                for basis_inds in list(itertools.permutations(list(range(0,mults[selected_orbit])),len(inds))):
                    print("basis_inds: {}".format(basis_inds))
                    print("orbit[selected_orbit]: \n{}\n".format(orbits[selected_orbit]))
                    tmp_array = []
                    for i in basis_inds:
                        tmp_array.append(orbits[selected_orbit][i])
                    b = np.array(tmp_array)
                    if np.allclose(b @ b.T,np.eye(b.shape[0])):
                        orthog_basis = b
                        print(" orthog_basis : \n{}".format(orthog_basis))
                        print(" orthog_basis @ orthog_basis.T: \n{}".format(orthog_basis @ orthog_basis.T))
                        break
                         
                if orthog_basis is None:
                    u,s,vh = np.linalg.svd(orbits[selected_orbit][0][:,None])
                    W = u[:,1:]
                    print("orbits[selected_orbit][0][:,None] :\n{}".format(orbits[selected_orbit][0][:,None]))
                    print("W : \n{}".format(W))
                    orthog_basis = np.hstack((orbits[selected_orbit][0][:,None],W))  
                    print(" orthog_basis : \n{}".format(orthog_basis))
                    print(" orthog_basis @ orthog_basis.T: \n{}".format(orthog_basis @ orthog_basis.T))
                
                Q_basis = orthog_basis.T @ Q[:,inds].T
                print("Q_basis : \n{}".format(Q_basis))
                for row in Q_basis:
                    newQ.append(row.T)
                print("newQ : \n{}".format(newQ))
                for v in orbits[selected_orbit]:
                    subspaces.append(v)

        newQ = np.array(newQ).T
        print("newQ: \n{}".format(newQ))
        print("newQ.T @ newQ\n{}".format(newQ.T @ newQ))
        return subspaces,newQ



            

    def find_all_subgroups(self):
        """ This algorithm will brute force enumerate subgroups of the full_group.

            A naive approach would be:
                Besides the identity, start deleting 1 by 1 sym ops and then    
                testing if the lefotver set of operations forms a group. 
            Maybe we can do better:
                Since these are going to be relatively simple point groups,
                if they have reflections, then
                they are giong to have equal number of rotations and reflections.
                1. Take only the rotations and brute force them.
                    ie remove operations 1 by 1, except the identity, and test 
                    if they form a group.

            Some observations:
                Lagrange's THM: every subgroup G_i of P, will have a number of elements
                    that divides the order of P. 
                    i.e. |P| % |G_i| = 0. 
                Every subgroup is a union of cyclic subgroups. 

                Together this means that the highest order is |P| / 2.
                So we can just   
                
            Best way:
               1. Enumerate cyclic subgroups. This means take each element and
                   multiply it by itelf until you get the indentity.
               2. Start forming unions of cyclic subgroups.
                    Say we have 4 cyclic subgroups, combine try all combinations   
                    which have a number of elements <= |P| / 2.
        """
        cyclic_subgroups = set()
        I = np.eye(self.dim)
        for  i, op in enumerate(self.full_rep):
            tmp_cyclic_subgroup = []
            tmp_op = op
            tmp_cyclic_subgroup.append(self.find_op(tmp_op))
            while not np.allclose(tmp_op,I,atol=self.tol):
                tmp_op = tmp_op @ op
                tmp_cyclic_subgroup.append(self.find_op(tmp_op))
                #print("tmp_op :\n{}\n".format(tmp_op))
            cyclic_subgroups.add(tuple(sorted(tmp_cyclic_subgroup)))
        print(" All cyclic subgroup:\n{}".format(cyclic_subgroups))
        subgroups = set()
        for num_cyc_sgs in range(1,len(self.full_rep)):
            cyc_sg_combos = list(itertools.combinations(cyclic_subgroups, num_cyc_sgs))
            for combo in cyc_sg_combos:
                test_sg_inds = set(itertools.chain.from_iterable(combo))
                test_group = [ self.full_rep[j] for j in test_sg_inds]
                testSymRep = SymRep(test_group)
                if (testSymRep.is_group()) and (len(test_group)<len(self.full_rep)):
                    #print("test_sg_inds is a group: {}".format(test_sg_inds))
                    subgroups.add(tuple(test_sg_inds))
                #else:
                #    print("test_sg_inds is NOT a group: {}".format(test_sg_inds))
        print(" All subgroups: \n{}\n".format(subgroups))
        return subgroups

            

    def find_op(self,test_op): 
        for i, op in enumerate(self.full_rep):
            if np.allclose(op,test_op,atol=self.tol):
                return i

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
        # is this right?
        # tA = np.mean([ op.T @ A @ op for op in self.full_rep ], axis=0)
        # below from A.1 JCT nonlinear elasticity def of invariant A matrix
        tA = np.mean([ op @ A @ op.T for op in self.full_rep ], axis=0)
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

    
RTOL = 1E-4
ATOL = 1E-7
TOL = 1E-6


         
def unique_vectors_from_list(vec_list):
    uvec_list = []
    for i,iv in enumerate(vec_list):
        if i==0:
            uvec_list.append(iv)
            continue
        in_list = False
        for j,jv in enumerate(uvec_list):
            if np.allclose(jv, iv, rtol=RTOL, atol=ATOL) or  \
                np.allclose(jv, -1.*iv, rtol=RTOL, atol=ATOL):
            #if np.allclose(jv, iv, rtol=RTOL, atol=ATOL):
                in_list = True
        if not in_list:
            uvec_list.append(iv)
    return uvec_list
def count_unique_vectors(vec_list, unique_vec_list):
    """ determines the number of times each element of 
        unique_vec_list occurs in vec_list  
        
    Returns:
        counters: counts in same order as unique_vec_list
    """

    counters = []
    for uvec in unique_vec_list:
        tmp_counter = 0
        for vec in vec_list:
            # if np.allclose(uvec, vec, atol=ATOL, rtol=RTOL):
            if np.allclose(uvec, vec, atol=ATOL, rtol=RTOL) or \
                np.allclose(uvec, -vec, atol=ATOL, rtol=RTOL):
                tmp_counter += 1
        counters.append(tmp_counter)
    return counters


def get_all_right_handed_2d_bases_from_orbit(o):
    """ Given list of high sym vectors find right handed basis 

    note: only 2d
    parameters:
        o : list of np.arrays
    returns:
        list of 2d np.array
    """
    right_handed_bases = []
    for (vi, vj) in itertools.permutations(o, r=2):
        # rows are basis vecs
        B = np.array([vi,vj])
        is_orthog = misc.is_zero_matrix(B @ B.T - np.eye(*B.shape))
        is_in_rhb = False
        for rhb in right_handed_bases:
            if not misc.is_zero_matrix(rhb - B):
                is_in_rhb = True

        if is_orthog and not is_in_rhb:
            right_handed_bases.append(B)
    return right_handed_bases



