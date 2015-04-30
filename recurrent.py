import logging

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import Linear, Softmax, Tanh
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.lookup import LookupTable
from blocks.extensions import Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from fuel.transformers import Batch
from fuel.schemes import ConstantScheme
from theano import tensor

from datastream import get_vocabulary, get_ngram_stream

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def construct_model(vocab_size, embedding_dim, ngram_order, hidden_dim,
                    activation):
                        
    # Construct the model
    x = tensor.lmatrix('features')
    y = tensor.lvector('targets')

    lookup = LookupTable(length=vocab_size, dim=embedding_dim, name='lookup')
    
    linear = Linear(input_dim=embedding_dim, output_dim=hidden_dim, name="linear")
    hidden = SimpleRecurrent(hidden_dim, activation)
    top_linear = Linear(input_dim=hidden_dim, output_dim=vocab_size, name="top_linear")

    embeddings = lookup.apply(x) # Return 3D Tensor: Batch X Time X embedding_dim
    embeddings = embeddings.dimshuffle(1, 0, 2)  # Give time as the first index: Time X Batch X embedding_dim
    
    pre_recurrent = linear.apply(embeddings)
    after_recurrent = hidden.apply(pre_recurrent)[-1] # It considers only the last hidden state
    presoft = top_linear.apply(after_recurrent)
    
    cost = Softmax().categorical_cross_entropy(y, presoft)

    # Initialize parameters
    for brick in (lookup, linear, hidden, top_linear):
        brick.weights_init = IsotropicGaussian(0.01)
        brick.biases_init = Constant(0.)
        brick.initialize()

    return cost


def train_model(cost, train_stream, valid_stream):
    cost.name = 'nll'
    perplexity = 2 ** (cost / tensor.log(2))
    perplexity.name = 'ppl'
    model = Model(cost)
    cg = ComputationGraph(cost)
    algorithm = GradientDescent(cost=cost, step_rule=Scale(learning_rate=0.01),
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            DataStreamMonitoring([cost, perplexity], valid_stream,
                                 prefix='valid'),
            Printing()
        ]
    )
    main_loop.run()

if __name__ == "__main__":
    # Test
    cost = construct_model(50000, 256, 6, 200, Tanh())
    vocabulary = get_vocabulary(50000)
    train_stream = Batch(get_ngram_stream(6, 'training', [1], vocabulary),
                         iteration_scheme=ConstantScheme(64))
    valid_stream = Batch(get_ngram_stream(6, 'heldout', [1], vocabulary),
                         iteration_scheme=ConstantScheme(256))
    train_model(cost, train_stream, valid_stream)
