import os
from itertools import count

import numpy
from fuel import config
from fuel.datasets import IndexableDataset
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers.text import NGrams

path = os.path.join(config.data_path, 'PennTreebankCorpus')


def get_data(data_size, vocab_size):
    """Get a training set with the given number of tokens."""
    train = numpy.load(os.path.join(
        path, 'penntree_char_and_word.npz'))['train_words']
    valid = numpy.load(os.path.join(
        path, 'penntree_char_and_word.npz'))['valid_words']
    train = train[:data_size]
    assert numpy.unique(train).size >= vocab_size
    ordered_tokens = numpy.argsort(numpy.bincount(train))[::-1]
    index_mapping = {old: new for old, new in zip(ordered_tokens, count())}
    train = [index_mapping.get(old, 591) for old in train]
    valid = [index_mapping.get(old, 591) for old in valid]
    return train, valid, {i: j for i, j in zip(count(), numpy.bincount(train))}


def get_ngram_stream(ngram_order, train):
    dataset = IndexableDataset({'features': [train]})
    stream = DataStream(dataset, iteration_scheme=SequentialExampleScheme(1))
    n_gram_stream = NGrams(ngram_order, stream)
    return n_gram_stream
