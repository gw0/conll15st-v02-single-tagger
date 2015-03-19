#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0326,W0621
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
import re

from gensim.models import word2vec

import common
log = common.logging.getLogger(__name__)


### PDTB parses

class PDTBParsesCorpus(object):
    """Iterate over sentences from the PDTB parses corpus."""

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in fileinput.input(self.fname):
            parses_dict = json.loads(line)
            log.debug("- loaded next PDTB parses line (size {})".format(len(line)))

            for doc_id in parses_dict:
                word_id = 0  # word offset within document
                sentence_id = 0  # sentence offset within document

                for sentence_dict in parses_dict[doc_id]['sentences']:
                    sentence_word_id = word_id

                    sentence = []
                    for word in sentence_dict['words']:
                        sentence.append({
                            'Text': word[0],
                            'DocID': doc_id,
                            'TokenList': [word_id],
                            'SentenceID': sentence_id,
                            'SentenceToken': sentence_word_id,
                        })
                        word_id += 1

                    yield sentence
                    sentence_id += 1


### Word2vec model

@common.profile
def load_word2vec(word2vec_bin):
    """Load a pre-trained word2vec model."""

    model = word2vec.Word2Vec.load_word2vec_format(word2vec_bin, binary=True)
    return model

@common.profile
@common.cache
def build_map_vocab(vocab, sentences, longest=True, max_len=3, delimiter="_"):
    """Build mapping of words/phrases in sentences to a vocabulary."""

    map_vocab = {}
    missing = []
    total_cnt = 0
    for sentence in sentences:
        for i in range(len(sentence)):
            total_cnt += 1
            if total_cnt % 10000:
                log.debug("- mapping {}, {}...".format(total_cnt, i))

            for j in range(min(i + max_len, len(sentence)), i, -1):
                text_list = [ sentence[k]['Text']  for k in range(i, j) ]

                # direct match
                text = delimiter.join(text_list)
                if text in vocab:
                    map_vocab[text] = text
                if text in map_vocab:
                    if longest:
                        break
                    else:
                        continue

                # optional underscores and strip numbers
                text_u = (delimiter + "?").join([ re.escape(text)  for text in text_list ])
                text_un = re.sub("[0-9][0-9]", "##", text_u)
                text_un = re.sub("##[0-9]", "###", text_un)
                pat_un = re.compile(text_un)

                for token in vocab:
                    if pat_un.match(token):
                        map_vocab[text] = token
                        break
                if text in map_vocab:
                    if longest:
                        break
                    else:
                        continue

                # ignore case, strip non-alphanumeric, optional underscores, strip numbers
                text2_u = (delimiter + "?").join([ re.sub("[^A-Za-z0-9#]", ".?", text)  for text in text_list ])
                text2_un = re.sub("[0-9][0-9]", "##", text2_u)
                text2_un = re.sub("##[0-9]", "###", text2_un)
                pat2_un = re.compile(text2_un, flags=re.IGNORECASE)

                for token in vocab:
                    if pat2_un.match(token):
                        map_vocab[text] = token
                        break
                if text in map_vocab:
                    if longest:
                        break
                    else:
                        continue

            if text not in map_vocab:
                missing.append(text)
                log.debug("- not in word2vec: {}".format(text))

    return map_vocab, missing, total_cnt

@common.profile
@common.cache
def build_map_word2vec(map_vocab, model):
    """Build mapping of words/phrases to word2vec vectors."""

    map_word2vec = {}
    for text, token in map_vocab.iteritems():
        map_word2vec[text] = model[token]  # raw numpy vector of a word
    return map_word2vec


if __name__ == '__main__':
    model_dir = sys.argv[1]
    parses_json = sys.argv[2].split(",")
    word2vec_bin = sys.argv[3]
    map_vocab_dump = "{}/map_vocab.dump".format(model_dir)
    map_word2vec_dump = "{}/map_word2vec.dump".format(model_dir)

    log.info("Loading word2vec model '{}'...".format(word2vec_bin))
    model = load_word2vec(word2vec_bin)

    log.info("Mapping words/phrases from {}...".format(parses_json))
    map_vocab, missing, total_cnt = build_map_vocab(model.vocab, PDTBParsesCorpus(parses_json))
    log.info("- mappings: {}, missing: {}, total words: {}".format(len(map_vocab), len(missing), total_cnt))

    log.info("Mapping words/phrases to word2vec vectors...")
    map_word2vec = build_map_word2vec(map_vocab, model)
    log.info("- words: {}".format(len(map_word2vec)))

    log.info("Save mappings to '{}'...".format(model_dir))
    import os
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    import joblib
    joblib.dump(map_vocab, map_vocab_dump)
    joblib.dump(map_word2vec, map_word2vec_dump)
