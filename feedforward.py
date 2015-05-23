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
from fuel.transformers import Batch
from fuel.schemes import ConstantScheme
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
    y = tensor.lvector('targets')

    lookup = LookupTable(length=vocab_size, dim=embedding_dim, name='lookup')
    hidden = MLP(activations=activations + [None],
                 dims=[ngram_order * embedding_dim] + hidden_dims +
                 [vocab_size])

    embeddings = lookup.apply(x)
    embeddings = embeddings.flatten(ndim=2)  # Concatenate embeddings
    activations = hidden.apply(embeddings)
    y_hat = Softmax().apply(activations)
    cost = Softmax().categorical_cross_entropy(y, activations)

    # Initialize parameters
    lookup.weights_init = IsotropicGaussian(0.001)
    hidden.weights_init = IsotropicGaussian(0.01)
    hidden.biases_init = Constant(0.001)
    lookup.initialize()
    hidden.initialize()

    return y, y_hat, cost


if __name__ == "__main__":
    # Test
    vocab_size = int(os.environ.get('VOCAB_SIZE', 10000))
    train_size = int(os.environ.get('TRAIN_SIZE', 929589))
    minibatch_size = 512
    num_batches = 100000 / minibatch_size

    y, y_hat, cost = construct_model(vocab_size, 512, 6, [256],
                                     [Rectifier()])
    cg = ComputationGraph([y, y_hat, cost])
    y, y_hat, cost = apply_dropout(
        cg, VariableFilter(roles=[INPUT], bricks=[Linear])(cg.variables), 0.5
    ).outputs
    train, valid, id_to_freq_mapping = get_data(train_size, vocab_size)

    # Make variational
    if len(sys.argv) > 1 and sys.argv[1] == 'variational':
        logger.info('Using the variational model')
        cost, sigmas = make_variational_model(cost)
    else:
        sigmas = None

    # Create monitoring channel
    freq_likelihood = FrequencyLikelihood(id_to_freq_mapping,
                                          requires=[y, y_hat],
                                          name='freq_costs')

    # Build training and validation datasets
    train_stream = Batch(get_ngram_stream(6, train),
                         iteration_scheme=ConstantScheme(minibatch_size))
    valid_stream = Batch(get_ngram_stream(6, valid),
                         iteration_scheme=ConstantScheme(minibatch_size))

    # Train
    sys.stdout = sys.stderr
    print('Hello world?')
    train_model(cost, train_stream, valid_stream, freq_likelihood,
                sigmas=sigmas, num_batches=num_batches,
                load_location=None,
                save_location='feedforward_{}_{}'.format(vocab_size,
                                                         train_size))
