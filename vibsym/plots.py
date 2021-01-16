import numpy as np
import typing as T
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

logger = logging.getLogger(__name__)

class PointCloud:
    def __init__(self,
                 init_state=None,
                 size=0.04,
                 freq=5,
                 Q=None):
        self.init_state = np.array(init_state)
        self.Q = Q
        self.shape = self.init_state.shape
        self.state = self.init_state.copy()
        self.size = size
        self.freq = freq
        self.time_elapsed = 0

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt

        # update positions
        self.state = self.init_state + self.Q * np.sin(
            -self.freq * self.time_elapsed)


def plot_normal_modes(coords: np.ndarray, Q: np.ndarray, dims: T.Sequence,
                      fps: float = 1. / 30):
    modes = np.arange(Q.shape[1])
    init_state = np.array(coords)
    boxes = []
    for mode in modes:
        q = np.reshape(Q[:, mode], init_state.shape)
        boxes.append(PointCloud(init_state, Q=q))

    dt = 1. / 30  # 30fps
    nqs = Q.shape[1]
    fig, axes = plt.subplots(1, Q.shape[1], figsize=(nqs, nqs // 2))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # particles holds the locations of the particles
    particles = []
    for i, ax in enumerate(axes):
        for j in range(dims[i] - 1):
            next(ax._get_lines.prop_cycler)

    for i, ax in enumerate(axes):
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        l, = ax.plot([], [], 'o', markersize=1)
        particles.append(l)
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    def init():
        """initialize animation"""
        for p in particles:
            p.set_data([], [])
        return particles
    def animate(i):
        """perform animation step"""
        for box in boxes:
            box.step(dt)

        ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
                 / np.diff(ax.get_xbound())[0])
        ms = 5

        # update pieces of the animation
        for p, box in zip(particles, boxes):
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
