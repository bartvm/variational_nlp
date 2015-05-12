import logging
from collections import defaultdict
import numpy
from collections import OrderedDict

from blocks.algorithms import GradientDescent, Scale, StepRule, CompositeRule
from blocks.bricks import Rectifier, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.filter import VariableFilter
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.roles import OUTPUT
from blocks.roles import add_role, ParameterRole
from blocks.utils import shared_floatx
from fuel.transformers import Batch
from fuel.schemes import ConstantScheme
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


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


class VarianceRole(ParameterRole):
    pass


VARIANCE = VarianceRole()


def make_variational_model(cost):
    # Consider the weights to be the means, create variances
    cg = ComputationGraph(cost)
    sigmas = {}
    for param in cg.parameters:
        sigmas[param] = shared_floatx(numpy.ones_like(param.get_value()) * 0.1,
                                      name=param.name + '_sigma')
        add_role(sigmas[param], VARIANCE)

    # Replace weights with samples from Gaussian
    rng = MRG_RandomStreams()
    new_cg = cg.replace({param: rng.normal(param.shape, param, sigmas[param])
                         for param in cg.parameters})
    return new_cg.outputs[0], sigmas


class VariationalInference(StepRule):
    def __init__(self, cost, sigmas, num_batches):
        self.cost = cost
        self.sigmas = sigmas
        self.num_batches = num_batches

    def compute_steps(self, previous_steps):
        # previous_steps contains parameters and their gradients
        params = self.sigmas.keys()

        # Create mu and sigma for prior, and their updates
        mu, sigma = \
            shared_floatx(0, name='mu'), shared_floatx(0.1, name='sigma')
        N = numpy.array(sum([param.get_value().size for param in params]),
                        dtype='float32')  # Else mean_param is float64
        mean_param = tensor.sum([param.sum() for param in params]) / N
        update_mu = (mu, mean_param)
        update_sigma = (sigma, tensor.sum([tensor.sum(self.sigmas[param] +
                        tensor.sqr(param - mean_param)) for param in params]) /
                        N)

        # Update parameters using gradient + regularization
        steps = OrderedDict(
            [(param, (param - mu) / sigma / self.num_batches +
              previous_steps[param])
             for param in params])

        # Update variance based on cost
        # NOTE: Sigma is actually sigma^2
        sigma_error_losses = {param: 0.5 * tensor.sqr(grad)
                              for param, grad in previous_steps.items()}
        steps.update(OrderedDict([(
            self.sigmas[param],
            0.5 * (1 / sigma - 1 / self.sigmas[param]) /
            self.num_batches +
            sigma_error_losses[param]) for param in params]))

        # For monitoring
        update_mu[1].name = 'prior_mu'
        update_sigma[1].name = 'prior_sigma'
        sigma = (tensor.sum([tensor.sum(_)
                             for _ in dict(self.sigmas).values()]) / N)
        sigma.name = 'posterior_sigmas'
        # dsigma = (tensor.sum([tensor.sum(_)
        #                       for _ in dict(update_sigmas).values()]) / N)
        # dsigma.name = 'posterior_sigma_delta'
        dsigma_c = (tensor.sum([tensor.sum(_)
                                for _ in sigma_error_losses.values()]) / N)
        dsigma_c.name = 'posterior_sigma_complexity_delta'
        dmu = (tensor.sum([tensor.sum(_) for _ in dict(steps).values()]) / N)
        dmu.name = 'posterior_mu_delta'
        # self.monitors = [update_mu[1], update_sigma[1], dsigma, dsigma_c, dmu

        return steps, [update_mu, update_sigma]


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

    step_rule = CompositeRule([VariationalInference(cg.outputs[0], sigmas,
                                                    1.2e5),
                               Scale(learning_rate=0.01)])
    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            DataStreamMonitoring([freq_likelihood, cost, perplexity],
                                 valid_stream,
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
            self.summed_likelihood[freq] += -numpy.log(prediction[target])
            self.total_seen[freq] += 1

    def readout(self):
        scores = numpy.zeros((len(self.summed_likelihood), 3))
        for i, freq in enumerate(sorted(self.summed_likelihood.keys())):
            scores[i] = [freq, self.total_seen[freq],
                         self.summed_likelihood[freq] / self.total_seen[freq]]
        return scores


if __name__ == "__main__":
    # Test
    cost = construct_model(50000, 256, 6, [128], [Rectifier()])
    vocabulary, id_to_freq_mapping = get_vocabulary(50000, True)

    # Make variational
    cost, sigmas = make_variational_model(cost)

    # Build training and validation datasets
    train_stream = Batch(get_ngram_stream(6, 'training', [1], vocabulary),
                         iteration_scheme=ConstantScheme(512))
    valid_stream = Batch(get_ngram_stream(6, 'heldout', [1], vocabulary),
                         iteration_scheme=ConstantScheme(256))

    # Train
    train_model(cost, train_stream, valid_stream, id_to_freq_mapping,
                load_location=None, save_location="trained_feedforward")
