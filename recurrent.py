import logging

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
from fuel.transformers import Batch, Padding, Mapping
from fuel.schemes import ConstantScheme
from theano import tensor


from datastream import (get_vocabulary, get_sentence_stream, frequencies,
                        add_frequency_mask, add_frequency_all)


logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

def construct_model(vocab_size, embedding_dim, hidden_dim,
                    activation):

    # Construct the model
    x = tensor.lmatrix('features')
    x_mask = tensor.fmatrix('features_mask')
    y = tensor.lmatrix('targets')
    # Batch X Time
    y_mask = tensor.fmatrix('targets_mask')
    # Batch X Time
    frequency_mask = tensor.fmatrix('frequency_mask')
    frequency_mask_mask = tensor.fmatrix('frequency_mask_mask')

    # Only for the validation
    last_word = tensor.lvector('last_word')

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
    # Give y as a vector and reshape presoft to 2D tensor
    y = y.flatten()

    shape = presoft.shape
    presoft = presoft.dimshuffle(1, 0, 2)
    presoft = presoft.reshape((shape[0] * shape[1], shape[2]))

    # Build cost_matrix
    presoft = presoft - presoft.max(axis=1).dimshuffle(0, 'x')
    log_prob = presoft - \
        tensor.log(tensor.exp(presoft).sum(axis=1).dimshuffle(0, 'x'))
    flat_log_prob = log_prob.flatten()
    range_ = tensor.arange(y.shape[0])
    flat_indices = y + range_ * presoft.shape[1]
    cost_matrix = flat_log_prob[flat_indices]

    # Mask useless values from the cost_matrix
    cost_matrix = - cost_matrix * \
        y_mask.flatten() * frequency_mask.flatten() * \
        frequency_mask_mask.flatten()

    # Average the cost
    cost = cost_matrix.sum()
    cost = cost / (y_mask * frequency_mask).sum()

    # Initialize parameters
    for brick in (lookup, linear, hidden, top_linear):
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.)
        brick.initialize()

    return cost


def train_model(cost, train_stream, valid_stream, valid_freq, valid_rare,
                load_location=None,
                save_location=None):
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
                                 prefix='valid', every_n_batches=500),
            DataStreamMonitoring([cost, perplexity], valid_rare,
                                 prefix='valid_rare', every_n_batches=500),
            DataStreamMonitoring([cost, perplexity], valid_freq,
                                 prefix='valid_frequent', every_n_batches=500),
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

if __name__ == "__main__":
    # Test
    cost = construct_model(50000, 256, 100, Tanh())
    vocabulary = get_vocabulary(50000)
    rare, freq = frequencies(vocabulary, 2000, 100)

    # Build training and validation datasets
    train_stream = Padding(Batch(Mapping(get_sentence_stream('training', [1], vocabulary),
                                         add_frequency_all, add_sources=("frequency_mask",)),
                                 iteration_scheme=ConstantScheme(64)))

    valid_stream = Padding(Batch(Mapping(get_sentence_stream('heldout', [1], vocabulary),
                                         add_frequency_all, add_sources=("frequency_mask",)),
                                 iteration_scheme=ConstantScheme(256)))

    valid_freq = Padding(Batch(Mapping(get_sentence_stream('heldout', [1], vocabulary),
                                       add_frequency_mask(freq), add_sources=("frequency_mask",)),
                               iteration_scheme=ConstantScheme(256)))

    valid_rare = Padding(Batch(Mapping(get_sentence_stream('heldout', [1], vocabulary),
                                       add_frequency_mask(rare), add_sources=("frequency_mask",)),
                               iteration_scheme=ConstantScheme(256)))

    # Train
    train_model(cost, train_stream, valid_stream, valid_freq, valid_rare,
                load_location=None,
                save_location=None)
