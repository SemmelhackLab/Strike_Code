from matplotlib import rcParams, animation
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


class BasePlotting(object):
    """Base class for setting defaults to plotting and saving figures.

    Parameters
    ----------
    kwargs: dict
        Passed to rcParams for changing defaults to plotting.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            rcParams[key] = value

    @staticmethod
    def save_figure(fig, output_path):
        """Saves figures with tight layout.

        Parameters
        ----------
        fig : plt.Figure
            Figure to be saved
        output_path : str
            Path for saving figure
        """
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        fig.savefig(output_path, dpi=300)


class AnimatedPlot(object):

    def __init__(self, animation_frames, fps=100):

        self.fig = None
        self.animation_frames = animation_frames
        self.fps = fps

    def animate(self, i):
        pass

    def play(self):
        ani = animation.FuncAnimation(self.fig, self.animate, self.animation_frames, interval=int(1000 / self.fps), blit=False)
        plt.show()

    def save(self, fname):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.fps, bitrate=4000)
        ani = animation.FuncAnimation(self.fig, self.animate, self.animation_frames, interval=int(1000 / self.fps), blit=False)
        ani.save(fname+'.mp4', writer=writer)

    @staticmethod
    def show():
        plt.show()


def plot_tail_kinematics(ax, k, fs=500., **kwargs):
    # Make sure variables are correct type
    k = np.array(k)
    fs = float(fs)
    # Unpack kwargs
    bout_length = len(k) / fs
    t_lim = kwargs.get('t_lim', (0, bout_length))
    first_frame = int(t_lim[0] * fs)
    last_frame = int(t_lim[1] * fs)
    k_max = kwargs.get('k_max', np.abs(k).max())
    ax_lim = kwargs.get('ax_lim', t_lim)
    # Select data to display
    k_display = k[first_frame:last_frame]
    # Plot data
    ax.imshow(k_display.T, aspect='auto', extent=(t_lim[0], t_lim[1], 0, 1), origin='lower', cmap='RdBu_r',
              clim=(-k_max, k_max), interpolation='bilinear')
    # Adjust axes
    ax.set_xlim(ax_lim)
    ax.set_ylim(1, 0)
    ax.axis('off')


def plot_trajectory(ax, t, fs=500., **kwargs):
    # Check projection type
    if (kwargs.get('projection', '2d') == '3d') or ax.name == '3d':
        projection_3d = True
    else:
        projection_3d = False
    # Make sure variables are correct type and shape
    if projection_3d:
        t = np.array(t[:, :3])
    else:
        t = np.array(t[:, :2])
    fs = float(fs)
    # Calculate portion of trajectory to plot
    bout_length = len(t) / fs
    t_lim = kwargs.get('t_lim', (0, bout_length))
    first_frame = int(t_lim[0] * fs)
    last_frame = int(t_lim[1] * fs)
    t_display = t[first_frame:last_frame]
    # Get the color scheme
    color = kwargs.get('color', 'time')
    lw = kwargs.get('lw', 3)
    if color.lower() in ['time', 't']:
        # Plot as line segments
        ps = np.expand_dims(t_display, 1)
        segments = np.concatenate([ps[:-1], ps[1:]], axis=1)
        # Normalize the color scheme
        c_lim = kwargs.get('c_lim', t_lim)
        cmap = kwargs.get('cmap', 'viridis')
        colormap = plt.cm.ScalarMappable(norm=Normalize(*c_lim), cmap=cmap)
        colors = colormap.to_rgba(np.linspace(t_lim[0], t_lim[1], len(segments)))
        # Plot line segments
        if projection_3d:
            lc = Line3DCollection(segments, colors=colors, lw=lw)
        else:
            lc = LineCollection(segments, colors=colors, lw=lw)
        ax.add_collection(lc)
    else:
        ax.plot(*t_display.T, c=color, lw=lw)
    # Adjust axes
    ax_lims = np.ceil(np.abs(t).max(axis=0))
    x_lim = kwargs.get('x_lim', (-ax_lims[0], ax_lims[0]))
    ax.set_xlim(x_lim)
    y_lim = kwargs.get('y_lim', (-ax_lims[1], ax_lims[1]))
    ax.set_ylim(y_lim)
    if projection_3d:
        z_lim = kwargs.get('z_lim', (-ax_lims[2], ax_lims[2]))
        ax.set_zlim(z_lim)


def plot_tail_series(ax, k, fs=500., **kwargs):
    k = np.array(k)
    fs = float(fs)
    # Calculate portion of trajectory to plot
    bout_length = len(k) / fs
    t_lim = kwargs.get('t_lim', (0, bout_length))
    first_frame = int(t_lim[0] * fs)
    last_frame = int(t_lim[1] * fs)
    k_display = k[first_frame:last_frame]
    # Get the color scheme
    color = kwargs.get('color', 'time')
    lw = kwargs.get('lw', 3)
    if color.lower() in ['time', 't']:
        # Normalize the color scheme
        c_lim = np.array(kwargs.get('c_lim', t_lim)) * fs
        cmap = kwargs.get('cmap', 'viridis')
        colormap = plt.cm.ScalarMappable(norm=Normalize(*c_lim), cmap=cmap)
        colors = colormap.to_rgba(np.arange(first_frame, last_frame))
    else:
        colors = [color] * len(k_display)

    theta = kwargs.get('rotation', 0)
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    for i, tail_shape in enumerate(k_display):
        vs = np.array([np.cos(tail_shape), np.sin(tail_shape)]).T
        ps = np.zeros((len(vs) + 1, 2))
        ps[1:] = np.cumsum(vs, axis=0)
        if kwargs.get('rotate', True):
            ps = np.dot(ps, R)

        ax.plot(*ps.T, color=colors[i], lw=lw)


def plot_explained_variance(ax, pca, **kwargs):

    n_components = kwargs.get('n_components', len(pca.components_))

    xpos = np.arange(0.5, n_components + 0.5)
    explained_variance = pca.explained_variance_ratio_[:n_components]
    cumulative_explained_variance = np.cumsum(explained_variance)

    ax.bar(xpos, explained_variance, width=1, color='r', edgecolor='k', zorder=0)
    ax.plot(xpos, cumulative_explained_variance, c='k', zorder=0)
    ax.scatter(xpos, cumulative_explained_variance, c='k', zorder=1)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim(0, n_components)
    ax.set_xticks(np.arange(n_components) + 0.5)
    ax.set_xticklabels(np.arange(1, n_components + 1))
    ax.tick_params(axis='x', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
