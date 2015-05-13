import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt

# Load the log
with open("trained_feedforward2/log", 'rb') as f:
    log = cPickle.load(f)

# Get the validation costs from the log
valid_costs = [val['valid_freq_costs']
               for key, val in log.items() if 'valid_freq_costs' in val]

# Extract the necessary axis
np.save('costs.npy', np.asarray(valid_costs)[:, 1:, 2])
Z = np.load('costs.npy')

freq = np.asarray(valid_costs)[0, 1:, 0]

# Diagonal plot


def diagonal():
    plt.plot(np.log(freq), Z[-1], np.log(freq), np.ones((freq.shape[0])) * 10.82)
    plt.ylim([0, 18])

# Rainbow plot
# The convolve part smooths the lines by taking a running average over 25 steps
# The rest is just for the colors
# The 1593 refers to the number of different word frequencies that are in
# the validation set


def rainbow(smooth):
    for line, color in zip(Z.T, plt.cm.ScalarMappable(plt.Normalize(0, 1593), plt.get_cmap('Spectral')).to_rgba(np.arange(1593))):
        plt.plot(np.convolve(line, np.ones(smooth) / (smooth * 1.))
                 [smooth:-smooth], c=color)
        #plt.plot(line, c=color)
    plt.ylim([0, 25])


if __name__ == "__main__":
    diagonal()
    # rainbow(25)
    plt.show()
