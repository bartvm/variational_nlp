import logging
import sys

from blocks.bricks import Rectifier, MLP, Softmax
from blocks.bricks.lookup import LookupTable
from blocks.initialization import IsotropicGaussian, Constant
from fuel.transformers import Batch
from fuel.schemes import ConstantScheme
from theano import tensor


from datastream import get_vocabulary, get_ngram_stream
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

    return x, y, y_hat, cost


if __name__ == "__main__":
    # Test
    vocab_size = 50000
    x, y, y_hat, cost = construct_model(vocab_size, 256, 6, [128],
                                        [Rectifier()])
    vocabulary, id_to_freq_mapping = get_vocabulary(vocab_size, True)

    # Make variational
    if len(sys.argv) > 1 and sys.argv[1] == 'variational':
        cost, sigmas = make_variational_model(cost)
    else:
        sigmas = None

    # Create monitoring channel
    freq_likelihood = FrequencyLikelihood(id_to_freq_mapping,
                                          requires=[y, y_hat],
                                          name='freq_costs')

    # Build training and validation datasets
    train_stream = Batch(get_ngram_stream(6, 'training', [1], vocabulary),
                         iteration_scheme=ConstantScheme(512))
    valid_stream = Batch(get_ngram_stream(6, 'heldout', [1], vocabulary),
                         iteration_scheme=ConstantScheme(512))

    # Train
    train_model(cost, train_stream, valid_stream, freq_likelihood,
                sigmas=sigmas, save_location="trained_feedforward")
