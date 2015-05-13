import numpy as np
from six.moves import cPickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Load the log
with open("C://Users/Eloi/Desktop/trained_feedforward/log", 'rb') as f:
    log = cPickle.load(f)

# Get the validation costs from the log
valid_costs = [val['valid_freq_costs']
               for key, val in log.items() if 'valid_freq_costs' in val]

# Extract the necessary axis
np.save('costs.npy', np.asarray(valid_costs)[:, :, 2])
Z = np.load('costs.npy')

freq = np.asarray(valid_costs)[0, :, 0]

# Diagonal plot


def diagonal():
    plt.plot(np.log(freq), Z[-1], np.log(freq),
             np.ones((freq.shape[0])) * 10.82)
    plt.ylim([0, 18])
    plt.show()
# Rainbow plot
# The convolve part smooths the lines by taking a running average over 25 steps
# The rest is just for the colors
# The 1593 refers to the number of different word frequencies that are in
# the validation set


def dynamic(speed):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(ylim=(0, 18), xlim=(0, 15))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = np.log(freq)
        y = Z[i*speed]
        line.set_data(x, y)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=Z.shape[0] / speed, interval=300, blit=True)

    plt.show()


def rainbow(smooth):
    for line, color in zip(Z.T, plt.cm.ScalarMappable(plt.Normalize(0, 1593), plt.get_cmap('Spectral')).to_rgba(np.arange(1593))):
        plt.plot(np.convolve(line, np.ones(smooth) / (smooth * 1.))
                 [smooth:-smooth], c=color)
        #plt.plot(line, c=color)
    plt.ylim([0, 25])
    plt.show()


if __name__ == "__main__":
    # diagonal()
    dynamic(1)
    # rainbow(25)
