import os
from collections import OrderedDict
from itertools import count

import numpy
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers.text import NGrams


def get_vocabulary(vocab_size, id_to_freq=False):
    assert vocab_size <= 10000
    path = os.path.join(config.data_path, 'PennTreebankCorpus')
    orig_tokens = numpy.load(os.path.join(path,
                                          'dictionaries.npz'))['unique_words']

    # Sort tokens by frequency
    train = numpy.load(os.path.join(
        path, 'penntree_char_and_word.npz'))['train_words']
    tokens = orig_tokens[numpy.argsort(numpy.bincount(train))[::-1]]
    vocabulary = OrderedDict(zip(tokens, range(vocab_size)))
    if id_to_freq:
        id_to_freq_mapping = {}
        train = numpy.load(os.path.join(
            path, 'penntree_char_and_word.npz'))['train_words']
        token_count = numpy.bincount([vocabulary[orig_tokens[token]]
                                      for token in train
                                      if orig_tokens[token] in vocabulary])
        id_to_freq_mapping = dict(zip(count(), token_count))
        return vocabulary, id_to_freq_mapping
    return vocabulary


def get_ngram_stream(ngram_order, which_set, which_partitions, vocabulary):
    path = os.path.join(config.data_path, 'PennTreebankCorpus')
    train = numpy.load(os.path.join(
        path, 'penntree_char_and_word.npz'))['{}_words'.format(which_set)]
    tokens = numpy.load(os.path.join(path, 'dictionaries.npz'))['unique_words']
    train = [vocabulary.get(tokens[token], 591) for token in train]
    dataset = IndexableDataset({'features': [train]})
    stream = DataStream(dataset, iteration_scheme=SequentialExampleScheme(1))
    n_gram_stream = NGrams(ngram_order, stream)
    return n_gram_stream