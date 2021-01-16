import vibsym as vs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#------------------------------------------------------------
np.set_printoptions(precision=14,suppress=True)
mol = vs.example_reps.get_example
# find sym ops from coordinates
# rep = vs.SymRep(coordinates=init_state)
# just build ops directly
rep = vs.SymRep(vs.repgen.generate_2D_cartesian_representation(permutations,rotations,reflections))
print(" is REP a group: {}".format(rep.is_group()))
trans_rota_Q = vs.repgen.trans_rota_basis_2D(init_state)
# for col_index in range(trans_rota_Q.shape[1]):
#     fig,ax = plt.subplots(1)
#     print("trans_rota\n{}".format(trans_rota_Q))
#     for ss,p in zip(trans_rota_Q[:,col_index].reshape(trans_rota_Q.shape[0]//2,2),init_state):
#         print(f'p:\n{p}')
#         print(f'ss:\n{ss}')
#         ax.quiver(p[0],p[1],ss[0],ss[1],scale=6)
#         ax.scatter(p[0],p[1])
#     ax.set_xlim([-2,2])
#     ax.set_ylim([-2,2])
#     plt.tight_layout()
#     plt.savefig(f'trans_rota_subspace_{col_index}.png')
#     plt.show()
print('trQ  = {}'.format(trans_rota_Q))
#exit()
print(' is big rep a group? {}'.format(rep.is_group()))
Q,dims = rep.block_diagonalize(trans_rota_Q)
print(f'Q:\n{Q}')
print(f'dims:\n{dims}')
# Q = np.append(Q,trans_rota_Q[:,-1][:,None],1)
# dims.append(max(dims)+1)
print("dims : {}".format(dims))
find_high_sym = True
if find_high_sym:
    subspaces,newQ = rep.find_high_symmetry_directions(Q,dims)
    Q = newQ
    #subgroups = rep.find_all_subgroups()
    #print(" all subgroups:\n {}".format(subgroups))
    fig,ax = plt.subplots(1)
    print("subspaces\n{}".format(subspaces))
    for ss in subspaces:
        if ss.shape[0]>1:
            for row in ss.T:

                #dot = np.dot(ss[:,0]/np.linalg.norm(ss[:,0]),ss[:,1]/np.linalg.norm(ss[:,1]))
                #print(" dot product = {}".format(dot))
                ax.quiver([0],[0],row[0],row[1],scale=4)
                #ax.plot(ss[0,:],ss[1,:])
    plt.savefig('high_sym.png')
    plt.show()


print("Q:\n{}".format(Q))
print("Q.T @ Q :\n{}".format(Q.T @ Q ))

class Molecule:
    def __init__(self,
                 init_state = [[1, 0, 0, -1],
                               [-0.5, 0.5, 0.5, 0.5],
                               [-0.5, -0.5, -0.5, 0.5]],
                 bounds = [-2, 2, -2, 2],
                 size = 0.04,
                 freq = 5,
                 Q = None):
        self.init_state =np.array(init_state) 
        self.Q =q 
        self.shape = self.init_state.shape
        self.state = self.init_state.copy()
        self.size=size
        self.freq= freq
        self.time_elapsed = 0

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt
        
        # update positions
        self.state =  self.init_state + self.Q*np.sin(-self.freq*self.time_elapsed)

modes = np.arange(Q.shape[1])

init_state = np.array(init_state)
boxes = []
for mode in modes: 
    q = np.reshape(Q[:,mode],init_state.shape)
    boxes.append(Molecule(init_state, Q=q))

dt = 1. / 30 # 30fps
#------------------------------------------------------------
# set up figure and animation
#fig = plt.figure()
nqs = Q.shape[1]
fig, axes = plt.subplots(1, Q.shape[1],figsize=(nqs,nqs//2))
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# particles holds the locations of the particles
particles = []
for i,ax in enumerate(axes):
    for j in range(dims[i]-1):
        next(ax._get_lines.prop_cycler)

for i,ax in enumerate(axes):
    #ax.xaxis.set_ticklabels([])
    #ax.yaxis.set_ticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    l, = ax.plot([], [], 'o', markersize=1)
    particles.append(l)
    ax.set_aspect('equal')
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)


def init():
    """initialize animation"""
    global box
    for p in particles:
        p.set_data([], [])
    return particles

def animate(i):
    """perform animation step"""
    global box, dt, ax, fig
    for box in boxes:
        box.step(dt)

    ms= int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    ms=5
    
    # update pieces of the animation
    for p,box in zip(particles,boxes):
        p.set_data(box.state[:, 0], box.state[:, 1])
        p.set_markersize(ms)
    return particles

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)
ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=interval, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# ani.save('normal_modes.mp4', dpi=200,fps=30, extra_args=['-vcodec', 'libx264'])
plt.tight_layout()
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('normal_modes.mp4', writer=Writer, dpi=100)
# ani.save('normal_modes.gif', fps=120,writer='imagemagick')
# ani.save('normal_modes.gif', writer='imagemagick')

plt.show()

