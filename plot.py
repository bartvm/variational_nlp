import numpy as np
from six.moves import cPickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Load the log
with open("C://Users/Eloi/Desktop/log_no_var_100_61_228406_682", 'rb') as f:
    log = cPickle.load(f)

# Get the validation costs from the log
valid_costs = [val['valid_freq_costs']
               for key, val in log.items() if 'valid_freq_costs' in val]

# Extract the necessary axis
np.save('costs.npy', np.asarray(valid_costs)[:, :, 2])
Z = np.load('costs.npy')
print Z.shape
freq = np.asarray(valid_costs)[0, :, 0]


# Load the log
with open("C://Users/Eloi/Desktop/log_recurrent_var_35_131000_708", 'rb') as f:
    log2 = cPickle.load(f)

# Get the validation costs from the log
valid_costs2 = [val['valid_freq_costs']
               for key, val in log2.items() if 'valid_freq_costs' in val]

valid_nll2 = [val['valid_nll'] for key, val in log2.items() if 'valid_nll' in val]
valid_nll2 = np.asarray(valid_nll2)[:]
imax = 0
for i in range(valid_nll2.shape[0]):
    if valid_nll2[i] > 7.09:
        imax = i

# Extract the necessary axis
np.save('costs2.npy', np.asarray(valid_costs2)[:, :, 2])
Z2 = np.load('costs2.npy')
print Z2.shape
freq2 = np.asarray(valid_costs2)[0, :, 0]


# Load the log
with open("C://Users/Eloi/Desktop/log_trained_recurrent_100_pretrained_var2_704_11_42000", 'rb') as f:
    log3 = cPickle.load(f)

# Get the validation costs from the log
valid_costs3 = [val['valid_freq_costs']
               for key, val in log3.items() if 'valid_freq_costs' in val]

valid_nll3 = [val['valid_nll'] for key, val in log3.items() if 'valid_nll' in val]
valid_nll3 = np.asarray(valid_nll3)[:]
imax3 = 0
for i in range(valid_nll3.shape[0]):
    print valid_nll3[i]
    # if valid_nll3[i] > 7.09:
    #     imax3 = i

# Extract the necessary axis
np.save('costs3.npy', np.asarray(valid_costs3)[:, :, 2])
Z3 = np.load('costs3.npy')
print Z3.shape
freq3 = np.asarray(valid_costs3)[0, :, 0]


# Diagonal plot
def diagonal():
    plt.plot(np.log(freq), Z[-1], 
            np.log(freq), np.ones((freq.shape[0])) * 10.82,
            np.log(freq2), Z2[-1], 
            np.log(freq3), Z3[-1],
             )
    plt.ylim([0, 18])
    plt.show()
# Rainbow plot
# The convolve part smooths the lines by taking a running average over 25 steps
# The rest is just for the colors
# The 1593 refers to the number of different word frequencies that are in
# the validation set


def dynamic(speed, Z=None, reference=None):
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
        line2.set_data(x, np.ones_like(Z[i]) * 10.82)
        line3.set_data(x, reference[-1])
        line4.set_data(x, Z2[i])
        return line, line2, line3, line4

    # call the animator.  blit=True means only re-draw the parts that have
    # changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=Z.shape[0], interval=speed, blit=True)
    plt.show()


def rainbow(smooth):
    for line, color in zip(Z.T, plt.cm.ScalarMappable(plt.Normalize(0, 1593), plt.get_cmap('Spectral')).to_rgba(np.arange(1593))):
        plt.plot(np.convolve(line, np.ones(smooth) / (smooth * 1.))
                 [smooth:-smooth], c=color)
        #plt.plot(line, c=color)
    plt.ylim([0, 25])
    plt.show()


if __name__ == "__main__":
    diagonal()
    dynamic(100, Z=Z3, reference=Z)
    # rainbow(8)
