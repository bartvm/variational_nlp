import logging
import os
from collections import OrderedDict
from itertools import chain, count, islice

from fuel import config
from fuel.datasets import OneBillionWord
from fuel.transformers.text import NGrams
from six import iteritems
from six.moves import cPickle, zip

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def get_vocabulary(vocab_size):
    # Load the word counts
    logger.info('Loading vocabulary')
    with open(os.path.join(config.data_path, '1-billion-word/processed/'
                           'one_billion_counter_full.pkl')) as f:
        word_counts = cPickle.load(f)

    # Construct the full vocabulary
    vocabulary = OrderedDict(zip(
        chain(['<S>', '</S>', '<UNK>'],
              (word for word, count in word_counts.most_common()
               if count >= 3)),
        count()))
    assert len(vocabulary) == 793471

    # Limit the vocabulary size
    if vocab_size is not None:
        vocabulary = OrderedDict(islice(vocabulary.items(), vocab_size))
    return vocabulary


def get_ngram_stream(ngram_order, which_set, which_partitions,
                     vocabulary):
    """Return an iterator over n-grams.

    Notes
    -----
    This reads the text files sequentially. However, note that the files are
    already shuffled.

    """

    # Construct data stream
    logger.info('Constructing data stream')
    dataset = OneBillionWord(which_set, which_partitions, vocabulary)
    data_stream = dataset.get_example_stream()
    n_gram_stream = NGrams(6, data_stream)

    return n_gram_stream


if __name__ == "__main__":
    # Test
    vocabulary = get_vocabulary(50000)
    stream = get_ngram_stream(6, 'training', range(1, 10), vocabulary)
    next(stream.get_epoch_iterator())
