import logging
import sys

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import Linear, Softmax, Tanh
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.lookup import LookupTable
from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import add_role, OutputRole
from fuel.transformers import Batch, Padding, Mapping
from fuel.schemes import ConstantScheme
from theano import tensor


from datastream import (get_vocabulary, get_sentence_stream)
from monitoring import FrequencyLikelihood
from variational import make_variational_model
from train import train_model

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def construct_model_r(vocab_size, embedding_dim, hidden_dim,
                      activation):

    # Construct the model
    #  All have shape (Batch, Time)
    x = tensor.lmatrix('features')
    x_mask = tensor.fmatrix('features_mask')
    y = tensor.lmatrix('targets')
    y_mask = tensor.fmatrix('targets_mask')

    lookup = LookupTable(length=vocab_size, dim=embedding_dim, name='lookup')
    linear = Linear(input_dim=embedding_dim, output_dim=hidden_dim,
                    name="linear")
    hidden = SimpleRecurrent(dim=hidden_dim, activation=activation,
                             name='hidden_recurrent')
    top_linear = Linear(input_dim=hidden_dim, output_dim=vocab_size,
                        name="top_linear")

    # Return 3D Tensor: Batch X Time X embedding_dim
    embeddings = lookup.apply(x)
    # Give time as the first index: Time X Batch X embedding_dim
    embeddings = embeddings.dimshuffle(1, 0, 2)
    pre_recurrent = linear.apply(embeddings)
    after_recurrent = hidden.apply(inputs=pre_recurrent,
                                   mask=x_mask.T)[:-1]
    after_recurrent_last = after_recurrent[-1]
    presoft = top_linear.apply(after_recurrent)

    # Define the cost
    # Compute the probability distribution
    time, batch, feat = presoft.shape
    presoft = presoft.dimshuffle(1, 0, 2)
    presoft = presoft.reshape((batch * time, feat))

    # Don't look at the prediction that had not enough words
    y_mask = tensor.set_subtensor(y_mask[:,:5],tensor.zeros_like(y_mask[:,:5]))

    y_hat = Softmax().apply(presoft)
    y = y.flatten()
    y_mask = y_mask.flatten()

    # Build cost_matrix
    distribution = presoft - presoft.max(axis=1).dimshuffle(0, 'x')
    log_prob = distribution - \
        tensor.log(tensor.exp(distribution).sum(axis=1).dimshuffle(0, 'x'))
    flat_log_prob = log_prob.flatten()
    range_ = tensor.arange(y.shape[0])
    flat_indices = y + range_ * distribution.shape[1]
    cost_matrix = - flat_log_prob[flat_indices]

    # Hide useless value in the cost_matrix
    cost_matrix = cost_matrix * y_mask

    # Average the cost
    cost = cost_matrix.sum()
    cost = cost / (y_mask.sum())
    add_role(cost, OutputRole())

    # Initialize parameters
    for brick in (lookup, linear, hidden, top_linear):
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.)
        brick.initialize()

    return y, y_hat, cost, y_mask


if __name__ == "__main__":
    # Test

    vocab_size = 50000
    max_sentence_length = 35
    minibatch_size = 64
    # B is the number of minibatches (formula 18)
    B = 237760 / (minibatch_size * 1.)

    y, y_hat, cost, y_mask = construct_model_r(50000, 256, 100, Tanh())
    vocabulary, id_to_freq_mapping = get_vocabulary(vocab_size, True)

    # Make variational
    if len(sys.argv) > 1 and sys.argv[1] == 'variational':
        logger.info('Using the variational model')
        cost, sigmas = make_variational_model(cost)
    else:
        sigmas = None

    # Create monitoring channel
    freq_likelihood = FrequencyLikelihood(id_to_freq_mapping,
                                          requires=[y, y_hat, y_mask],
                                          name='freq_costs')
    # Build training and validation datasets
    train_stream = Padding(Batch(get_sentence_stream('training', [1], vocabulary, max_sentence_length),
                                 iteration_scheme=ConstantScheme(minibatch_size)))

    valid_stream = Padding(Batch(get_sentence_stream('heldout', [1], vocabulary, max_sentence_length),
                                 iteration_scheme=ConstantScheme(minibatch_size)))

    # Train
    train_model(cost, train_stream, valid_stream, freq_likelihood,
                sigmas=sigmas, B=B,
                load_location=None,
                save_location="trained_recurrent_no_beginning_large_plot",
                learning_rate=0.01)
