import logging
import os
from collections import OrderedDict
from itertools import chain, count, islice

from fuel import config
from fuel.datasets import OneBillionWord
from fuel.transformers import Mapping, Filter
from fuel.transformers.text import NGrams
from six import iteritems
from six.moves import cPickle, zip

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def get_vocabulary(vocab_size, id_to_freq=False):
    # Load the word counts
    logger.info('Loading vocabulary')
    with open(os.path.join(config.data_path, '1-billion-word/processed/'
                           'one_billion_counter_00001.pkl')) as f:
        word_counts = cPickle.load(f)

    # Construct the full vocabulary
    vocabulary = OrderedDict(zip(
        chain(['<S>', '</S>', '<UNK>'],
              (word for word, count in word_counts.most_common()
               if count >= 3)),
        count()))
    # assert len(vocabulary) == 793471
    assert len(vocabulary) == 65181

    # Limit the vocabulary size
    if vocab_size is not None:
        vocabulary = OrderedDict(islice(vocabulary.items(), vocab_size))

    # Optional: Create a mapping from word ID to its frequency
    id_to_freq_mapping = {}
    if id_to_freq:
        for word, index in iteritems(vocabulary):
            id_to_freq_mapping[index] = word_counts[word]
        return vocabulary, id_to_freq_mapping

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
    n_gram_stream = NGrams(ngram_order, data_stream)

    return n_gram_stream


class _FilterLong(object):
    def __init__(self, max_length):
        self.max_length = max_length

    def __cal__(self, data):
        return len(data[0]) <= self.max_length


def _shift_words(sample):
    sentence = sample[0]
    result = sentence[1:]
    return (result,)


def get_sentence_stream(which_set, which_partitions, vocabulary,
                        max_length=50):
    """Return an iterator over sentences

    Notes
    -----
    This reads the text files sequentially. However, note that the files are
    already shuffled.

    """
    # Construct data stream
    logger.info('Constructing data stream')
    dataset = OneBillionWord(which_set, which_partitions, vocabulary)
    data_stream = dataset.get_example_stream()

    # Get rid of long sentences that don't fit
    data_stream = Filter(data_stream, _FilterLong(max_length))

    # Creates the dataset "targets"
    data_stream = Mapping(data_stream, _shift_words, add_sources=("targets",))

    return data_stream
