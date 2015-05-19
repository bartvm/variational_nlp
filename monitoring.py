from collections import defaultdict

import numpy
from blocks.monitoring.aggregation import MonitoredQuantity


class FrequencyLikelihood(MonitoredQuantity):

    def __init__(self, word_counts, **kwargs):
        """Calculate the likelihood as a function of word frequency.

        Parameters
        ----------
        word_counts : dict
            A dictionary mapping word indices (int) to their frequency
            (int).

        """
        super(FrequencyLikelihood, self).__init__(**kwargs)
        self.word_counts = word_counts

    def initialize(self):
        self.summed_likelihood = defaultdict(int)
        self.total_seen = defaultdict(int)

    def accumulate(self, targets, predictions, masks):
        for i, (target, prediction, mask) in enumerate(zip(targets, predictions, masks)):
            if mask:
                freq = self.word_counts[target]
                if freq:  # Skip <S>, </S> and <UNK>
                    self.summed_likelihood[freq] += - \
                        numpy.log(prediction[target])
                    self.total_seen[freq] += 1

    def readout(self):
        scores = numpy.zeros((len(self.summed_likelihood), 3))
        for i, freq in enumerate(sorted(self.summed_likelihood.keys())):
            scores[i] = [freq, self.total_seen[freq],
                         self.summed_likelihood[freq] / self.total_seen[freq]]
        self.initialize()
        return scores
