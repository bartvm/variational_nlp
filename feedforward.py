import logging
from collections import defaultdict
import numpy as np

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import Rectifier, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.roles import OUTPUT
from fuel.transformers import Batch
from fuel.schemes import ConstantScheme
from theano import tensor


from datastream import get_vocabulary, get_ngram_stream

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def construct_model(vocab_size, embedding_dim, ngram_order, hidden_dims,
                    activations):
    # Construct the model
    x = tensor.lmatrix('features')
    y = tensor.lvector('targets')

    lookup = LookupTable(length=vocab_size, dim=embedding_dim, name='lookup')
    hidden = MLP(activations=activations + [None],
                 dims=[ngram_order * embedding_dim] + hidden_dims +
                 [vocab_size])

    embeddings = lookup.apply(x)
    embeddings = embeddings.flatten(ndim=2)  # Concatenate embeddings
    activations = hidden.apply(embeddings)
    cost = Softmax().categorical_cross_entropy(y, activations)

    # Initialize parameters
    lookup.weights_init = IsotropicGaussian(0.001)
    hidden.weights_init = IsotropicGaussian(0.01)
    hidden.biases_init = Constant(0.001)
    lookup.initialize()
    hidden.initialize()

    return cost


def train_model(cost, train_stream, valid_stream, id_to_freq_mapping,
                load_location=None, save_location=None):
    cost.name = 'nll'
    perplexity = 2 ** (cost / tensor.log(2))
    perplexity.name = 'ppl'

    # Define the model
    model = Model(cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(cost)
    targets, = [var for var in cg.inputs if var.name == 'targets']
    activations, = VariableFilter(bricks=[MLP], roles=[OUTPUT])(cg.variables)
    predictions = Softmax().apply(activations)
    freq_likelihood = FrequencyLikelihood(id_to_freq_mapping,
                                          requires=[targets, predictions],
                                          name='freq_costs')

    algorithm = GradientDescent(cost=cost, step_rule=Scale(learning_rate=0.01),
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            DataStreamMonitoring([freq_likelihood], valid_stream,
                                 prefix='valid', every_n_batches=500),
            Printing(every_n_batches=500)
        ]
    )
    main_loop.run()

    # Save the main loop
    if save_location is not None:
        logger.info('Saving the main loop...')
        dump_manager = MainLoopDumpManager(save_location)
        dump_manager.dump(main_loop)
        logger.info('Saved')


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

    def accumulate(self, targets, predictions):
        for i, (target, prediction) in enumerate(zip(targets, predictions)):
            freq = self.word_counts[target]
            self.summed_likelihood[freq] += -np.log(prediction[target])
            self.total_seen[freq] += 1

    def readout(self):
        scores = np.zeros((len(self.summed_likelihood), 3))
        for i, freq in enumerate(sorted(self.summed_likelihood.keys())):
            scores[i] = [freq, self.total_seen[freq],
                         self.summed_likelihood[freq] / self.total_seen[freq]]
        return scores


if __name__ == "__main__":
    # Test
    cost = construct_model(50000, 256, 6, [128], [Rectifier()])
    vocabulary, id_to_freq_mapping = get_vocabulary(50000, True)

    # Build training and validation datasets
    train_stream = Batch(get_ngram_stream(6, 'training', [1], vocabulary),
                         iteration_scheme=ConstantScheme(64))
    valid_stream = Batch(get_ngram_stream(6, 'heldout', [1], vocabulary),
                         iteration_scheme=ConstantScheme(256))

    # Train
    train_model(cost, train_stream, valid_stream, id_to_freq_mapping,
                load_location=None, save_location="trained_feedforward")
