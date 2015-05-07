import logging
import numpy

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import Rectifier, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import add_role, VariableRole
from blocks.utils import shared_floatx
from fuel.transformers import Batch, Filter
from fuel.schemes import ConstantScheme
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


from datastream import (get_vocabulary, get_ngram_stream, frequencies,
                        FilterWords)

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


class VarianceRole(VariableRole):
    pass


VARIANCE = VarianceRole()


def make_variational(cost, cost_grads, learning_rate):
    # Consider the weights to be the means, create variances
    cg = ComputationGraph(cost)
    sigmas = {}
    for param in cg.parameters:
        sigmas[param] = shared_floatx(numpy.ones_like(param.get_value()))
        add_role(sigmas[param], VARIANCE)

    # Replace weights with samples from Gaussian
    rng = MRG_RandomStreams()
    new_cg = cg.replace({param: rng.normal(param.shape, param, sigmas[param])
                         for param in cg.parameters})

    # Create mu and sigma for prior, and their updates
    mu, sigma = shared_floatx(0), shared_floatx(1)
    N = tensor.sum([param.size for param in cg.parameters])
    mean_param = tensor.sum([param.sum() for param in new_cg.parameters]) / N
    update_mu = (mu, mean_param)
    update_sigma = (sigma, tensor.sum([tensor.sum(
        tensor.sqr(param - mean_param)) for param in new_cg.parameters]) / N)

    # Update variance based on cost
    # NOTE: Sigma is actually sigma^2
    sigma_error_losses = [0.5 * tensor.sqr(cost_grad)
                          for cost_grad in cost_grads]
    update_sigmas = [(sigmas[param],
                      -0.5 * (1 / sigma - 1 / sigmas[param]) -
                      sigma_error_losses) for param in new_cg.parameters]
    update_mus= [(param,
                 -(param - mu) / sigma ** 2 - cost_grads)
                 for param in new_cg.parameters]


def train_model(cost, train_stream, valid_stream, valid_freq, valid_rare,
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
    algorithm = GradientDescent(cost=cost, step_rule=Scale(learning_rate=0.01),
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            DataStreamMonitoring([cost, perplexity], valid_stream,
                                 prefix='valid_all', every_n_batches=5000),
            # Overfitting of rare words occurs between 3000 and 4000 iterations
            DataStreamMonitoring([cost, perplexity], valid_rare,
                                 prefix='valid_rare', every_n_batches=500),
            DataStreamMonitoring([cost, perplexity], valid_freq,
                                 prefix='valid_frequent',
                                 every_n_batches=5000),
            Printing(every_n_batches=500)
        ]
    )
    main_loop.run()

    #Save the main loop
    if save_location is not None:
        logger.info('Saving the main loop...')
        dump_manager = MainLoopDumpManager(save_location)
        dump_manager.dump(main_loop)
        logger.info('Saved')

if __name__ == "__main__":
    # Test
    cost = construct_model(50000, 256, 6, [128], [Rectifier()])
    vocabulary = get_vocabulary(50000)
    rare, frequent = frequencies(vocabulary, 200, 100)

    # Build training and validation datasets
    train_stream = Batch(get_ngram_stream(6, 'training', [1], vocabulary),
                         iteration_scheme=ConstantScheme(64))
    valid_stream = Batch(get_ngram_stream(6, 'heldout', [1], vocabulary),
                         iteration_scheme=ConstantScheme(256))

    filt_freq = FilterWords(frequent)
    filt_rare = FilterWords(rare)

    valid_freq = Batch(Filter(get_ngram_stream(6, 'heldout', [1], vocabulary),
                              filt_freq),
                       iteration_scheme=ConstantScheme(256))
    valid_rare = Batch(Filter(get_ngram_stream(6, 'heldout', [1], vocabulary),
                              filt_rare),
                       iteration_scheme=ConstantScheme(256))

    # Train
    train_model(cost, train_stream, valid_stream, valid_freq, valid_rare,
                load_location=None,
                save_location="trained_feedforward")
