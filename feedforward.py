import logging

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
from fuel.transformers import Batch, Filter
from fuel.schemes import ConstantScheme
from theano import tensor


from datastream import get_vocabulary, get_ngram_stream, frequencies, FilterWords

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
                                 prefix='valid_all', every_n_batches=5000),
            DataStreamMonitoring([cost, perplexity], valid_rare,
                                 prefix='valid_rare', every_n_batches=5000),
            DataStreamMonitoring([cost, perplexity], valid_freq,
                                 prefix='valid_frequent', every_n_batches=5000),
            Printing(every_n_batches=5000)
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
    cost = construct_model(50000, 256, 6, [128], [Rectifier()])
    vocabulary = get_vocabulary(50000)
    rare, frequent = frequencies(vocabulary, 200)
    
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
