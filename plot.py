import sys
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def diagonal(Z, freq, vocab_size, smooth=0, Z2=None):

    if smooth:
        smoothed = np.convolve(Z[-1], np.ones(smooth) / smooth)[smooth:-smooth]
        plt.plot(
            np.log(freq)[smooth // 2:len(smoothed) + smooth // 2], smoothed)
    else:
        plt.plot(np.log(freq), Z[-1])

    if Z2 is not None:
        if smooth:
            smoothed2 = np.convolve(
                Z2[-1], np.ones(smooth) / smooth)[smooth:-smooth]
            plt.plot(
                np.log(freq)[smooth // 2:len(smoothed) + smooth // 2], smoothed2)

        else:
            plt.plot(np.log(freq), Z2[-1])

    plt.plot(np.log(freq), np.ones((freq.shape[0])) * -np.log(1. / vocab_size))

    plt.ylim([0, 18])
    plt.xlim([0, 12])
    plt.show()


def dynamic(Z, freq, vocab_size, speed):

    # First set up the figure, the axis, and the plot element we want to
    # animate
    fig = plt.figure()
    ax = plt.axes(ylim=(0, 18), xlim=(0, 15))
    line, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)
    line3, = ax.plot([], [], lw=2)
    line4, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        return line, line2, line3, line4

    # animation function.  This is called sequentially
    def animate(i):
        x = np.log(freq)
        y = Z[i]
        line.set_data(x, y)
        line2.set_data(x, np.ones_like(Z[i]) * -np.log(1. / vocab_size))
        return line, line2

    # call the animator.  blit=True means only re-draw the parts that have
    # changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=Z.shape[0], interval=speed,
                                   blit=True)
    plt.show()


def rainbow(Z, freq, smooth):
    """Rainbow plot

    The convolve part smooths the lines by taking a running average over 25
    steps The rest is just for the colors The 1593 refers to the number of
    different word frequencies that are in the validation set.

    """
    for line, color in \
            zip(Z.T, plt.cm.ScalarMappable(plt.Normalize(0, len(Z.T)),
                                           plt.get_cmap('Spectral')).to_rgba(
            np.arange(len(Z.T)))):
        if smooth:
            plt.plot(np.convolve(line, np.ones(smooth) / (smooth * 1.))
                     [smooth:-smooth], c=color)
        else:
            plt.plot(line, c=color)
    plt.ylim([0, 25])
    plt.show()


if __name__ == "__main__":
    # Usage: plot.py log vocab_size [log_reference]
    # Example: plot.py trained_feedforward_var/log 50000
    # trained_feedforward/log
    with open(sys.argv[1], 'rb') as f:
        log = cPickle.load(f)

    # Get the validation costs from the log
    valid_costs = [val['valid_freq_costs']
                   for key, val in log.items() if 'valid_freq_costs' in val]
    # Extract the necessary axis
    Z = np.asarray(valid_costs)[:, :, 2]
    freq = np.asarray(valid_costs)[0, :, 0]

    smooth_diag = 0
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'rb') as f:
            log2 = cPickle.load(f)
        valid_costs2 = [val['valid_freq_costs']
                        for key, val in log2.items() if 'valid_freq_costs' in val]

        Z2 = np.asarray(valid_costs2)[:, :, 2]
        diagonal(Z, freq, int(sys.argv[2]), smooth_diag, Z2)
    else:
        diagonal(Z, freq, int(sys.argv[2]), smooth_diag)

    dynamic(Z, freq, int(sys.argv[2]), 100)
    rainbow(Z, freq, 0)
