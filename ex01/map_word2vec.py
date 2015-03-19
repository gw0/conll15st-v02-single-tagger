#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Map vocabulary to word2vec vectors for conll15st experiment 01.

Usage: ./map_word2vec.py <model_dir> <parses_json> <word2vec_bin>

  - <model_dir>: directory for storing mapping and other resources
  - <parses_json>: conll15st automatic PDTB parses in JSON format
  - <word2vec_bin>: existing word2vec model in C binary format

> ./map_word2vec.py ex01_model pdtb_trial_parses.json GoogleNews-vectors-negative300.bin.gz
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import sys
import fileinput
import json

from gensim.models import word2vec

import common
log = common.logging.getLogger(__name__)


### Vocabulary

def parses_iter_tokens(f):
    """Read all tokens from PDTB parses."""

    for line in f:
        parses_dict = json.loads(line)

        for doc_id in parses_dict:
            token_id = -1  # token offset within the document
            sentence_id = -1  # sentence offset within the document

            for sentence in parses_dict[doc_id]['sentences']:
                sentence_id += 1
                token_lid = -1  # token offset within the sentence

                for token in sentence['words']:
                    token_id += 1
                    token_lid += 1
                    char_begin = token[1]['CharacterOffsetBegin']
                    char_end = token[1]['CharacterOffsetEnd']
                    yield ([doc_id, char_begin, char_end, token_id, sentence_id, token_lid, token[0]])

@common.profile
@common.cache
def build_vocab(parses_json):
    """Build vocabulary with counters from PDTB parses."""

    f = fileinput.input(parses_json)

    vocab = {}
    for token in parses_iter_tokens(f):
        word = token[-1]
        try:
            vocab[word] += 1
        except KeyError:
            vocab[word] = 1
    return vocab


### Word2vec model

@common.profile
def load_word2vec(word2vec_bin):
    """Load a pre-trained word2vec model."""

    model = word2vec.Word2Vec.load_word2vec_format(word2vec_bin, binary=True)
    return model

@common.profile
@common.cache
def build_map_word2vec(vocab, model):
    """Map vocabulary to word2vec vectors."""

    map_word2vec = {}
    for word in vocab:
        try:
            map_word2vec[word] = model[word]  # raw numpy vector of a word
        except KeyError:
            log.debug("- not in word2vec: {} ({})".format(word, vocab[word]))
    return map_word2vec


if __name__ == '__main__':
    model_dir = sys.argv[1]
    parses_json = sys.argv[2].split(",")
    word2vec_bin = sys.argv[3]
    map_dump = "{}/map_word2vec.dump".format(model_dir)

    log.info("Building vocabulary from {}...".format(parses_json))
    vocab = build_vocab(parses_json)
    log.info("- words: {}, total count: {}".format(len(vocab), sum(vocab.itervalues())))

    log.info("Loading word2vec model '{}'...".format(word2vec_bin))
    model = load_word2vec(word2vec_bin)
    log.info("- words: {}, dims: {}".format(len(model.vocab), model.layer1_size))

    log.info("Mapping vocabulary to word2vec vectors...")
    map_word2vec = build_map_word2vec(vocab, model)
    log.info("- words: {}".format(len(map_word2vec)))

    log.info("Save mapping to '{}'...".format(map_dump))
    import os
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    import joblib
    joblib.dump(map_word2vec, map_dump)
