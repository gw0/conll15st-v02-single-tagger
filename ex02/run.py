#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Experiment 02 for CoNLL 2015 (Shallow Discourse Parsing).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import os
import joblib
import json
import numpy as np
import theano
import theano.tensor as T

import time
from collections import defaultdict
from collections import OrderedDict

import common
import data_pdtb
import data_word2vec
from scorer import scorer, validator
log = common.logging.getLogger(__name__)

# debugging
import socket
if socket.gethostname() == "elite":
    theano.config.optimizer = 'fast_compile'
    theano.config.exception_verbosity = 'high'


# similar to https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/rnnslu.py
class RNN_deep(object):

    def __init__(self, x_dim, hidden_dim, y_dim):

        # parameters of the model
        self.wx = theano.shared(name="wx", value=0.1 * np.random.uniform(-1.0, 1.0, (x_dim + hidden_dim + 1, hidden_dim)).astype(theano.config.floatX))
        self.hx_0 = theano.shared(name="hx_0", value=np.zeros(hidden_dim, dtype=theano.config.floatX))

        self.w1 = theano.shared(name="w1", value=0.1 * np.random.uniform(-1.0, 1.0, (hidden_dim + hidden_dim + 1, hidden_dim)).astype(theano.config.floatX))
        self.h1_0 = theano.shared(name="h1_0", value=np.zeros(hidden_dim, dtype=theano.config.floatX))

        self.wy = theano.shared(name="wy", value=0.1 * np.random.uniform(-1.0, 1.0, (hidden_dim + 1, y_dim)).astype(theano.config.floatX))

        # bundle
        self.params = [self.wx, self.hx_0, self.w1, self.h1_0, self.wy]

        # define recurrent neural network
        # (for each input word predict all output tags)
        x = T.fmatrix("x")
        y = T.fmatrix("y")
        learn_rate = T.fscalar('learn_rate')

        activation = T.tanh
        #activation = T.nnet.sigmoid
        #activation = lambda x: x * (x > 0)  # reLU
        #activation = lambda x: x * ((x > 0) + 0.01)
        #activation = lambda x: T.minimum(x * (x > 0), 6)  # capped reLU

        def recurrence(x_cur, hx_prev, h1_prev):
            hx = activation(T.dot(T.concatenate([x_cur, hx_prev, [1.0]]), self.wx))
            h1 = activation(T.dot(T.concatenate([hx, h1_prev, [1.0]]), self.w1))
            y_pred = activation(T.dot(T.concatenate([h1, [1.0]]), self.wy))
            return (hx, h1, y_pred)

        (hx, h1, y_pred), _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.hx_0, self.h1_0, None], n_steps=x.shape[0])

        #loss = lambda y_pred, y: T.mean((y_pred - y) ** 2)  # MSE
        #loss = lambda y_pred, y: T.sum((y_pred - y) ** 16) ** (1.0/16)
        #loss = lambda y_pred, y: T.max((y_pred - y) ** 2)
        loss = lambda y_pred, y: T.max(abs(y - y_pred)) + T.mean((y - y_pred) ** 2)
        #loss = lambda y_pred, y: T.sum((y_pred - y) ** 16) ** (1.0/16) + T.mean((y - y_pred) ** 2)
        l1_reg = 0.001
        l1 = T.mean(self.wx) + T.mean(self.w1) + T.mean(self.wy)
        l2_reg = 0.001
        l2 = T.mean(self.wx ** 2) + T.mean(self.w1 ** 2) + T.mean(self.wy ** 2)

        # define gradients and updates
        cost = loss(y_pred, y) + l1_reg * l1 + l2_reg * l2
        gradients = T.grad(cost, wrt=self.params)
        updates = OrderedDict((p, p - learn_rate * g)  for p, g in zip(self.params, gradients))

        # compile theano functions
        self.predict = theano.function(inputs=[x], outputs=y_pred)
        self.train = theano.function(inputs=[x, y, learn_rate], outputs=[cost, T.min(y_pred), T.max(y_pred
            ), T.mean(y_pred), y_pred], updates=updates)


def extract_relations(doc_id, doc, y, tag_to_j, all_tsenses, thres=0.5):
    # Find all relations above threshold

    for tsense in all_tsenses:
        rtype, sense = tsense.split(":")

        rnum = 0
        while rnum >= 0:
            relation = {}
            relation['DocID'] = doc_id
            relation['Type'] = rtype
            relation['Sense'] = [sense]

            relation_len = 0
            for part in ['Arg1', 'Arg2', 'Connective']:
                relation[part] = {'TokenList': []}
                if relation['Type'] == 'Implicit' and part == 'Connective':
                    continue

                tag = "{0}:{1}:{2}:{3}".format(rtype, sense, rnum, part)
                if tag not in tag_to_j:
                    rnum = -1
                    break

                # transform to token list
                j = tag_to_j[tag]
                token_list = {}
                for word, tags in zip(doc, y):
                    if tags[j] > thres:
                        for k in word['TokenList']:
                            token_list[k] = 1
                relation[part]['TokenList'] = sorted(token_list.keys())
                relation_len += len(relation[part]['TokenList'])

            if rnum < 0:
                break
            print tsense, len(relation['Arg1']['TokenList']), len(relation['Arg2']['TokenList']), len(relation['Connective']['TokenList'])
            if len(relation['Arg1']['TokenList']) > 2 and len(relation['Arg2']['TokenList']) > 2 and relation_len < 300:
                yield relation
            rnum += 1


def load(pdtb_dir, word2vec_bin, word2vec_dim, tag_to_j):
    """Load PDTB data and transform it to numerical form."""

    parses_ffmt = "{}/pdtb-parses.json"
    relations_ffmt = "{}/pdtb-data.json"
    raw_ffmt = "{}/raw/{}"

    # load relations
    relations = load_relations(relations_ffmt.format(pdtb_dir), tag_to_j)

    # prepare mapping vocabulary to word2vec vectors
    map_word2vec = joblib.load("./cache/map_word2vec.dump")

    # load words from PDTB parses
    words = load_words(pdtb_dir, relations)

    # prepare numeric form
    x = []
    y = []
    for doc_id, doc in words.iteritems():
        doc_x = []
        doc_y = []
        for word in doc:
            # map text to word2vec
            try:
                doc_x.append(map_word2vec[word['Text']])
            except KeyError:  # missing in vocab
                doc_x.append(np.zeros(word2vec_dim))

            # map tags to vector
            tags = [0.0] * len(tag_to_j)
            for tag, count in word['Tags'].iteritems():
                tags[tag_to_j[tag]] = float(count)
            doc_y.append(tags)

            #print word['Text'], word['Tags']
            #print word['Text'], doc_x[-1][0:1], doc_y[-1]

        x.append(np.asarray(doc_x, dtype=np.float32))
        y.append(np.asarray(doc_y, dtype=np.float32))

    return x, y, words, relations


def relation_to_tag(relation, rpart):
    """Convert relation to tag."""

    rtype = relation['Type']
    rsense = relation['Sense'][0]  # assume only first sense
    rnum = relation['SenseNum'][0]  # assume only first sense
    return ":".join([rtype, rsense, str(rnum), rpart])


def load_relations(relations_json, tag_to_j):
    """Load PDTB relations by document id.

    Example output:

        relations[doc_id][0] = {
            'Arg1': {'CharacterSpanList': [[2493, 2517]], 'RawText': 'and told ...', 'TokenList': [[2493, 2496, 465, 15, 8], [2497, 2501, 466, 15, 9], ...]},
            'Arg2': {'CharacterSpanList': [[2526, 2552]], 'RawText': "they're ...", 'TokenList': [[2526, 2530, 472, 15, 15], [2530, 2533, 473, 15, 16], ...]},
            'Connective': {'CharacterSpanList': [[2518, 2525]], 'RawText': 'because', 'TokenList': [[2518, 2525, 471, 15, 14]]},
            'TokenMin': 465,
            'TokenMax': 476,
            'TokenCount': 12,
            'DocID': 'wsj_1000',
            'ID': 15007,
            'Type': 'Explicit',
            'Sense': ['Contingency.Cause.Reason'],
            'SenseNum': [1],
        }
    """

    # load all relations
    f = open(relations_json, 'r')
    relations_all = {}
    for line in f:
        relation = json.loads(line)

        # fix inconsistent structure
        if 'TokenList' not in relation['Arg1']:
            relation['Arg1']['TokenList'] = []
        if 'TokenList' not in relation['Arg2']:
            relation['Arg2']['TokenList'] = []
        if 'TokenList' not in relation['Connective']:
            relation['Connective']['TokenList'] = []

        # add token id min and max and token count
        token_list = sum([ relation[part]['TokenList']  for part in ['Arg1', 'Arg2', 'Connective'] ], [])
        token_list = [ t[2]  for t in token_list ]  # from gold format to token ids
        relation['TokenMin'] = min(token_list)
        relation['TokenMax'] = max(token_list)
        relation['TokenCount'] = len(token_list)

        # store by document id
        try:
            relations_all[relation['DocID']].append(relation)
        except KeyError:
            relations_all[relation['DocID']] = [relation]
    f.close()

    # order and filter relations
    relations = {}
    for doc_id in relations_all:
        # order by increasing token count
        relations_all[doc_id].sort(key=lambda r: r['TokenCount'])

        # filter by specified tags
        relations[doc_id] = []
        rnums = {}
        for relation in relations_all[doc_id]:
            rnum_key = (relation['Type'], relation['Sense'][0])
            try:
                rnums[rnum_key] += 1
            except KeyError:
                rnums[rnum_key] = 1
            relation['SenseNum'] = [rnums[rnum_key]]

            for rpart in ['Arg1', 'Arg2', 'Connective']:
                if relation_to_tag(relation, rpart) in tag_to_j:  # relation found
                    relations[doc_id].append(relation)
                    break  # only one
    return relations


def load_words(pdtb_dir, relations):
    """Load PDTB words by document id.

    Example output:

        words[doc_id][0] = {
            'Text': "Kemper",
            'DocID': doc_id,
            'ParagraphID': 0,
            'SentenceID': 0,
            'SentenceToken': 0,
            'TokenList': [0],
            'PartOfSpeech': "NNP",
            'Linkers': ["arg1_14890"],
            'Tags': {"Explicit:Expansion.Conjunction:4:Arg1": 1},
        }
    """

    lpart_to_rpart = {"arg1": "Arg1", "arg2": "Arg2", "conn": "Connective"}
    words_it = data_pdtb.PDTBParsesCorpus(pdtb_dir, with_document=True, with_paragraph=False, with_sentence=False, word_split="-|\\\\/", word_meta=True)

    words = {}
    for doc in words_it:
        doc_id = doc[0]['DocID']

        # store by document id
        words[doc_id] = doc

        # add relation tags to each word
        for word in words[doc_id]:
            word['Tags'] = {}
            for linker in word['Linkers']:  # get relation ids for each word
                lpart, rid = linker.split("_")
                rpart = lpart_to_rpart[lpart]

                # find by relation id
                for relation in relations[doc_id]:
                    if rid == str(relation['ID']):  # relation found
                        tag = relation_to_tag(relation, rpart)
                        try:
                            word['Tags'][tag] += 1
                        except KeyError:
                            word['Tags'][tag] = 1
                        break  # only one
    return words


if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description="Run experiment 02 for CoNLL 2015 (Shallow Discourse Parsing).")
    argp.add_argument('model_dir',
        help="directory for storing trained model and other resources")
    argp.add_argument('train_dir',
        help="conll15st dataset directory for training")
    argp.add_argument('valid_dir',
        help="conll15st dataset directory for validation")
    argp.add_argument('test_dir',
        help="conll15st dataset directory for testing (only 'pdtb-parses.json')")
    argp.add_argument('output_dir',
        help="output directory for system predictions (in 'output.json')")
    args = argp.parse_args()

    # load data
    output_json = "{}/output.json".format(args.output_dir)
    word2vec_bin = "./GoogleNews-vectors-negative300.bin.gz"
    word2vec_dim = 300

    tag_to_j = {}
    tag_to_j["Explicit:Expansion.Conjunction:1:Arg1"] = len(tag_to_j)
    tag_to_j["Explicit:Expansion.Conjunction:1:Arg2"] = len(tag_to_j)
    tag_to_j["Explicit:Expansion.Conjunction:1:Connective"] = len(tag_to_j)

    x_train, y_train, train_words, train_relations = load(args.train_dir, word2vec_bin, word2vec_dim, tag_to_j)
    #x_valid, y_valid, valid_words, valid_relations = load(args.valid_dir, word2vec_bin, word2vec_dim, tag_to_j)
    #x_test, _, valid_words, _ = load(args.test_dir, word2vec_bin, word2vec_dim, tag_to_j)
    x_valid, y_valid, valid_words, valid_relations = x_train, y_train, train_words, train_relations
    x_test, _, valid_words, _ = x_train, y_train, train_words, train_relations

    #import copy
    #train_relations_out = copy.deepcopy(train_relations[doc_id])
    #for relation in train_relations_out:
    #    relation['Arg1']['TokenList'] = [ t[2]  for t in relation['Arg1']['TokenList'] ]
    #    relation['Arg2']['TokenList'] = [ t[2]  for t in relation['Arg2']['TokenList'] ]
    #    relation['Connective']['TokenList'] = [ t[2]  for t in relation['Connective']['TokenList'] ]


    # settings
    rand_seed = int(time.time())
    learn_rate = 0.1
    decay_after = 10
    decay_rate = 0.95
    decay_min = 1e-8
    epochs = 10000
    x_dim = 300
    hidden_dim = 30  #XXX: x_dim
    y_dim = len(tag_to_j)

    # instantiate the model
    print "rand_seed={0}".format(rand_seed)
    np.random.seed(rand_seed)
    rnn = RNN_deep(x_dim=x_dim, hidden_dim=hidden_dim, y_dim=y_dim)

    # iterate through train dataset
    best_cost = np.inf
    best_f1 = -np.inf
    best_epoch = 0
    epoch = 0
    while epoch < epochs or learn_rate > decay_min:

        # train model
        t = time.time()
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            cost, y_min, y_max, y_mean, y = rnn.train(x, y, np.array(learn_rate, dtype=np.float32))
            print "learning epoch {} ({:.2f}%) ({:.2f} sec), rate {:.2e}, train cost {}{}".format(epoch, (i + 1) * 100.0 / len(x_train), time.time() - t, learn_rate, cost, (" +" if cost < best_cost else ""))
            print y_min, y_max, y_mean
            if cost < best_cost:
                best_cost = cost
                best_epoch = epoch

        # predict with model
        y = rnn.predict(x_train[0])

        # extract relations
        doc_id = "wsj_1000"
        y_relations = []
        for relation in extract_relations(doc_id, train_words[doc_id], y, tag_to_j):
            y_relations.append(relation)

            # check relations
            validator.check_type(relation)    
            validator.check_sense(relation)
            validator.check_args(relation)
            validator.check_connective(relation)

        # evaluate relations
        try:
            precision, recall, f1 = scorer.evaluate_relation(train_relations[doc_id], y_relations)
            print "evaluate train set precision {0:.2f}, recall {1:.2f}, f1 {2:.2f}".format(precision, recall, f1)
        except ZeroDivisionError:
            precision, recall, f1 = 0.0, 0.0, 0.0
        if f1 == 1.0:
            print "WOOHOO!!!"
            break

        # learning rate decay if no improvement after some epochs
        if epoch - best_epoch >= decay_after:
            learn_rate *= decay_rate
        epoch += 1
