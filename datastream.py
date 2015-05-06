import logging
import os
from collections import OrderedDict
from itertools import chain, count, islice

from numpy import vectorize
from numpy.random import randint
from fuel import config
from fuel.datasets import OneBillionWord
from fuel.transformers import Mapping, Batch, Filter, Transformer
from fuel.transformers.text import NGrams
from six.moves import cPickle, zip

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


def get_vocabulary(vocab_size):
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
    return vocabulary


# Vocabulary is already sorted by decreasing frequency
def frequencies(vocabulary, nb_rare, nb_freq):
    frequent_words = OrderedDict(islice(vocabulary.items(), nb_freq))
    rare_words = OrderedDict(islice(OrderedDict(
        reversed(vocabulary.items())).items(), nb_rare))
    return rare_words, frequent_words


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


# Function applied by the filter, that determine whether the word is frequent
# is in the list.
class FilterWords(object):

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, batch):
        ngram, target = batch
        return target in self.dictionary.values()


def _filter_long(data):
    return len(data[0]) <= 100


def _shift_words(sample):
    sentence = sample[0]
    result = sentence[1:]
    return (result,)


def get_sentence_stream(which_set, which_partitions, vocabulary):
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
    data_stream = Filter(data_stream, _filter_long)

    # Creates the dataset "targets"
    data_stream = Mapping(data_stream, _shift_words, add_sources=("targets",))

    return data_stream


class add_frequency_mask(object):

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, batch):
        sentence, target = batch
        return (vectorize(lambda index: index in self.dictionary.values())(target),)

def add_frequency_all(batch):
    sentence, target = batch
    return (vectorize(lambda index: True)(target),)

if __name__ == "__main__":
    # Test
    vocabulary = get_vocabulary(50000)
    rare, frequent = frequencies(vocabulary, 5000, 100)
    stream = Mapping(get_sentence_stream('heldout', [1], vocabulary),
                     add_frequency_all, add_sources=("frequency_mask",))

    print next(stream.get_epoch_iterator())
    print next(stream.get_epoch_iterator())
