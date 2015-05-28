from __future__ import division

import logging
import os
import sys

from blocks.bricks import Rectifier, MLP, Softmax, Linear
from blocks.bricks.lookup import LookupTable
from blocks.graph import ComputationGraph, apply_dropout
from blocks.roles import INPUT
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant
from theano import tensor


from penntree import get_data, get_ngram_stream
from monitoring import FrequencyLikelihood
from variational import make_variational_model
from train import train_model

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def construct_model(vocab_size, embedding_dim, ngram_order, hidden_dims,
                    activations):
    # Construct the model
    x = tensor.lmatrix('features')
    y = tensor.lmatrix('targets')

    lookup = LookupTable(length=vocab_size, dim=embedding_dim, name='lookup')
    hidden = MLP(activations=activations + [None],
                 dims=[ngram_order * embedding_dim] + hidden_dims +
                 [vocab_size])

    embeddings = lookup.apply(x)
    embeddings = embeddings.flatten(ndim=2)  # Concatenate embeddings
    activations = hidden.apply(embeddings)
    y_hat = Softmax().apply(activations)
    cost = Softmax().categorical_cross_entropy(y.flatten(), activations)

    # Initialize parameters
    lookup.weights_init = IsotropicGaussian(0.001)
    hidden.weights_init = IsotropicGaussian(0.01)
    hidden.biases_init = Constant(0.001)
    lookup.initialize()
    hidden.initialize()

    return y, y_hat, cost


from blocks.dump import load_parameter_values
from blocks.model import Model
import numpy as np


def custom_init(cost, id_to_freq_mapping, train_size, vocab_size,
                load_location=None, dropped_cost=None):

    if dropped_cost is None:
        dropped_cost = cost

    # Define the model
    model = Model(dropped_cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    cg = ComputationGraph(dropped_cost)
    print(cg.parameters)

    b_init = np.zeros((vocab_size)).astype(np.float32)
    for i, val in enumerate(id_to_freq_mapping.values()):
        b_init[i] = val / (train_size * 1.)

    rval = np.log(b_init)
    rval -= rval.mean()
    cg.parameters[0].set_value(rval)

    return cost, dropped_cost


if __name__ == "__main__":
    # Test
    vocab_size = int(os.environ.get('VOCAB_SIZE', 10000))
    train_size = int(os.environ.get('TRAIN_SIZE', 929589))
    minibatch_size = 256
    num_batches = 100000 / minibatch_size

    y, y_hat, cost = construct_model(vocab_size, 512, 6, [256],
                                     [Rectifier()])

    cg = ComputationGraph([y_hat, cost])
    dropped_y_hat, dropped_cost = apply_dropout(
        cg, VariableFilter(roles=[INPUT], bricks=[Linear])(cg.variables), 0.5
    ).outputs
    train, valid, id_to_freq_mapping = get_data(train_size, vocab_size)

    # Make variational
    if len(sys.argv) > 1 and sys.argv[1] == 'variational':
        logger.info('Using the variational model')
        dropped_cost, sigmas = make_variational_model(dropped_cost)
    else:
        sigmas = None

    # Create monitoring channel
    freq_likelihood = FrequencyLikelihood(id_to_freq_mapping,
                                          requires=[y, y_hat],
                                          name='freq_costs')

    # Build training and validation datasets
    train_stream = get_ngram_stream(6, train, minibatch_size)
    valid_stream = get_ngram_stream(6, valid, minibatch_size)

    # Train
    sys.stdout = sys.stderr

    cost, dropped_cost = custom_init(cost, id_to_freq_mapping,
                                     train_size, vocab_size,
                                     load_location="params.npz",
                                     dropped_cost=dropped_cost)

    train_model(cost, train_stream, valid_stream, freq_likelihood,
                sigmas=sigmas, num_batches=num_batches,
                save_location='feedforward_{}_{}'.format(vocab_size,
                                                         train_size),
                dropped_cost=dropped_cost)
