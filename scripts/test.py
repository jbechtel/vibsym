#import vibnet as vn
import vibsym as vs
import numpy as np

plot="tri"
plot="sq"
if  plot=="tri":
    rep = vs.SymRep(vs.repgen.generate_2D_triangle_cartesian_representation())
    trans_rota_Q = vs.repgen.equilateral_2D_triangle_trans_and_rota()
if plot=="square":
    rep = vs.SymRep(vs.repgen.generate_2D_square_cartesian_representation())
    trans_rota_Q = vs.repgen.equilateral_2D_square_trans_and_rota()
print(' is rep a group? {}'.format(rep.is_group()))
Q = rep.block_diagonalize(trans_rota_Q)

#new_rep = [ Q.T @ op @ Q for op in rep.full_rep ]
#np.set_printoptions(precision=3,suppress=True)
#for op in new_rep:
#    print(op)
##

vs.plotvib.plot_disp_dofs_2D_equil_triangle(Q)
