#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0326,W0621
"""
Map vocabulary to word2vec vectors for conll15st experiment 01.

Usage: ./map_word2vec.py <model_dir> <word2vec_bin> <parses_json>

  - <model_dir>: directory for storing mapping and other resources
  - <word2vec_bin>: existing word2vec model in C binary format
  - <parses_json>: conll15st automatic PDTB parses in JSON format

> ./map_word2vec.py ex01_model GoogleNews-vectors-negative300.bin.gz pdtb_trial_parses.json
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


### Vocabulary mapping

def strip_phrase(text, _pats=None):
    """Strip phrase to basic lowercase alphanumeric characters."""
    if _pats is None:
        _pats = [
            (re.compile("[^a-z0-9#\\.,;/%$]"), ""),
            (re.compile("[0-9]+"), "#"),
            #(re.compile("[0-9][0-9]"), "##"),
            #(re.compile("##[0-9]"), "###"),
        ]

    strip = text.lower()
    for pat, repl in _pats:
        strip = pat.sub(repl, strip)
    return strip


@common.profile
def map_strip_base(vocab_base, get_count=None):
    """Build mapping of stripped helper vocabulary to base vocabulary."""
    if get_count is None:
        get_count = lambda vocab, token: vocab[token].count

    vocab_strip = {}
    for token in vocab_base:
        strip = strip_phrase(token)
        if strip not in vocab_strip:
            vocab_strip[strip] = token
        else:
            cnt_cur = get_count(vocab_base, vocab_strip[strip])
            cnt_new = get_count(vocab_base, token)
            if cnt_new > cnt_cur:
                vocab_strip[strip] = token
    return vocab_strip

@common.profile
def map_sent_base(sentences, vocab_base, vocab_strip=None, longest=True, max_len=5, delimiter="_"):
    """Build mapping of words/phrases in sentences to base vocabulary."""
    if vocab_strip is None:
        vocab_strip = {}

    pat = re.compile("[^a-z0-9#\\.,;/%$]")

    vocab_sent = {}
    missing = {}
    cnt = 0
    for sentence in sentences:
        for i in range(len(sentence)):
            if cnt % 50000 == 0:
                log.debug("- processing at {}...".format(cnt))
            cnt += 1

            for j in range(min(i + max_len, len(sentence)), i, -1):
                # already matched
                text_list = [ sentence[k]['Text']  for k in range(i, j) ]
                text = delimiter.join(text_list)
                if text in vocab_sent:
                    break

                # direct match
                if text in vocab_base:
                    vocab_sent[text] = text
                    if longest:
                        break
                    else:
                        continue

                # phrase check
                if pat.sub("", text_list[0]) == "" or pat.sub("", text_list[-1]) == "":
                    continue

                # lower case, strip non-alphanumeric, strip numbers
                strip = strip_phrase(text)
                if strip in vocab_strip:
                    vocab_sent[text] = vocab_strip[strip]
                    if longest:
                        break
                    else:
                        continue

            if text not in vocab_sent:
                try:
                    missing[text] += 1
                except KeyError:
                    missing[text] = 1
                    log.debug("- not in word2vec: {}".format(text))
    return vocab_sent, missing, cnt

### PDTB parses

class PDTBParsesCorpus(object):
    """Iterate over sentences from the PDTB parses corpus."""

    def __init__(self, fname, reg_split=None):
        self.fname = fname
        self.reg_split = reg_split

    def __iter__(self):
        for line in fileinput.input(self.fname):
            log.debug("- loading PDTB parses line (size {})".format(len(line)))
            parses_dict = json.loads(line)

            for doc_id in parses_dict:
                token_id = 0  # token offset within document
                sentence_id = 0  # sentence offset within document

                for sentence_dict in parses_dict[doc_id]['sentences']:
                    sentence_token_id = token_id

                    sentence = []
                    for token in sentence_dict['words']:
                        for word in re.split(self.reg_split, token[0]):
                            sentence.append({
                                'Text': word,
                                'DocID': doc_id,
                                'TokenList': [token_id],
                                'SentenceID': sentence_id,
                                'SentenceToken': sentence_token_id,
                            })
                        token_id += 1

                    yield sentence
                    sentence_id += 1


### Word2vec model

@common.profile
def load_word2vec(word2vec_bin):
    """Load a pre-trained word2vec model."""

    model = word2vec.Word2Vec.load_word2vec_format(word2vec_bin, binary=True)
    return model

@common.profile
def map_base_word2vec(vocab_base, model):
    """Build mapping of vocabulary to word2vec vectors."""

    map_word2vec = {}
    cnt = 0
    for text, token in vocab_base.iteritems():
        if cnt % 50000 == 0:
            log.debug("- processing at {}...".format(cnt))
        cnt += 1

        map_word2vec[text] = model[token]  # raw numpy vector of a word
    return map_word2vec


### General

@common.cache
def build(word2vec_bin, parses_json):
    log.info("Loading word2vec model '{}'...".format(word2vec_bin))
    model = load_word2vec(word2vec_bin)

    log.info("Mapping stripped helper vocabulary...")
    model_strip = map_strip_base(model.vocab)

    log.info("Mapping words/phrases from {}...".format(parses_json))
    sentences_iter = PDTBParsesCorpus(parses_json, reg_split="-|\\\\/")
    vocab, missing, total_cnt = map_sent_base(sentences_iter, model.vocab, model_strip, longest=True)
    log.info("- mappings: {}, missing: {}, total words: {}".format(len(vocab), len(missing), total_cnt))

    log.info("Mapping vocabulary to word2vec vectors...")
    map_word2vec = map_base_word2vec(vocab, model)
    log.info("- words: {}".format(len(map_word2vec)))

    return map_word2vec, vocab, missing


if __name__ == '__main__':
    model_dir = sys.argv[1]
    word2vec_bin = sys.argv[2]
    parses_json = sys.argv[3].split(",")
    map_word2vec_dump = "{}/map_word2vec.dump".format(model_dir)
    vocab_dump = "{}/map_word2vec_vocab.dump".format(model_dir)
    missing_dump = "{}/map_word2vec_missing.dump".format(model_dir)

    log.info("Building...")
    map_word2vec, vocab, missing = build(word2vec_bin, parses_json)

    log.info("Saving to '{}'...".format(model_dir))
    import os
    for dump in [map_word2vec_dump, vocab_dump, missing_dump]:
        dump_dir = os.path.dirname(dump)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
    import joblib
    joblib.dump(map_word2vec, map_word2vec_dump, compress=1)
    joblib.dump(vocab, vocab_dump, compress=1)
    joblib.dump(missing, missing_dump, compress=1)

#map_word2vec = joblib.load("ex01_model/map_word2vec.dump")
#vocab = joblib.load("ex01_model/map_word2vec_vocab.dump")
#missing = joblib.load("ex01_model/map_word2vec_missing.dump")
