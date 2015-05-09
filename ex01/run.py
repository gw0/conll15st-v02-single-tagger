#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Experiment 01 for CoNLL 2015 (Shallow Discourse Parsing).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import os
import json
import numpy as np
import theano
import theano.tensor as T

import time
from collections import OrderedDict

import common
import data_word2vec
import data_pdtb
from scorer import scorer, validator
log = common.logging.getLogger(__name__)

# debugging
import socket
if socket.gethostname() == "elite" and False:
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


def extract_relation2(doc_id, doc, y, j_to_tag, subtract=1.0):
    # Find max relation, nearest contiguous spans

    max_i, max_j, max_probab = find_max(doc, y)
    print max_i, max_j, max_probab
    tag_split = j_to_tag[max_j].split(":")

    relation = {}
    relation['DocID'] = doc_id
    relation['Type'] = tag_split[0]
    relation['Sense'] = [tag_split[1]]

    for part in ['Arg1', 'Arg2', 'Connective']:
        relation[part] = {'TokenList': []}
        if relation['Type'] == 'Implicit' and part == 'Connective':
            continue

        tag_split[-1] = part
        j = tag_to_j[":".join(tag_split)]
        i_b, i_b_diff = find_nearest_before(y, max_i, j, max_probab, 1.0)
        i_a, i_a_diff = find_nearest_after(y, max_i, j, max_probab, 1.0)
        if i_b_diff < i_a_diff:
            i = i_b
        else:
            i = i_a
        if i is None:
            print "no nearest", max_i, tag_split, j
            return None

        probab = y[i][j]
        i_min, i_max = find_largest_continous(y, i, j, probab, 0.5)

        # subtract probability
        for i in range(i_min, i_max + 1):
            y[i][j] -= subtract

        # transform to token list
        token_list = {}
        for i in range(i_min, i_max + 1):
            for k in doc[i]['TokenList']:
                token_list[k] = 1
        relation[part]['TokenList'] = sorted(token_list.keys())

    return relation


def find_max(doc, y):
    # find max probable tag
    max_probab = -1
    max_i = -1
    max_j = -1
    for i, (word, tags) in enumerate(zip(doc, y)):
        #print i, word['Text'], tags
        for j, tag_probab in enumerate(tags):
            if tag_probab > max_probab:
                max_probab = tag_probab
                max_i = i
                max_j = j
    return (max_i, max_j, max_probab)

def find_nearest_before(y, i, j, probab, diff):
    thres = probab - diff
    if thres < 0.01:
        thres = 0.01

    # find nearest tag to given position
    i_min = i
    while i_min >= 0 and y[i_min][j] < thres:
        i_min -= 1

    if i_min < 0:
        return (None, len(y))
    return (i_min, i - i_min)

def find_nearest_after(y, i, j, probab, diff):
    thres = probab - diff
    if thres < 0.01:
        thres = 0.01

    # find nearest tag to given position
    i_max = i
    while i_max < len(y) and y[i_max][j] < thres:
        i_max += 1

    if i_max >= len(y):
        return (None, len(y))
    return (i_max, i_max - i)

def find_largest_continous(y, i, j, probab, diff):
    thres = probab - diff
    if thres < 0.0:
        thres = 0.0

    # largest continuous text span
    i_min = i
    while i_min >= 0 and y[i_min][j] > thres:
        i_min -= 1
    i_min += 1

    i_max = i
    while i_max < len(y) and y[i_max][j] > thres:
        i_max += 1
    i_max -= 1

    print j_to_tag[j], " ".join([ doc[i]['Text']  for i in range(i_min, i_max + 1) ])
    return (i_min, i_max)


if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description="Run experiment 01 for CoNLL 2015 (Shallow Discourse Parsing).")
    argp.add_argument('train_dir',
        help="conll15st dataset directory for training")
    argp.add_argument('model_dir',
        help="directory for storing trained model and other resources")
    argp.add_argument('valid_dir',
        help="conll15st dataset directory for validation")
    argp.add_argument('pred_dir',
        help="conll15st dataset directory for prediction (only 'pdtb-parses.json')")
    argp.add_argument('output_dir',
        help="output directory for system predictions")
    args = argp.parse_args()

    # prepare mapping vocabulary to word2vec vectors
    word2vec_dump = "{}/map_word2vec.dump".format(args.model_dir)
    word2vec_bin = "./GoogleNews-vectors-negative300.bin.gz"
    if not os.path.exists(word2vec_dump):
        map_word2vec, _, _ = data_word2vec.build(word2vec_bin, [args.train_dir, args.valid_dir, args.pred_dir])
    else:
        import joblib
        map_word2vec = joblib.load(word2vec_dump)

    # load data
    train_x_json = "{}/pdtb-parses.json".format(args.train_dir)
    train_y_json = "{}/pdtb-data.json".format(args.train_dir)
    valid_x_json = "{}/pdtb-parses.json".format(args.valid_dir)
    valid_y_json = "{}/pdtb-data.json".format(args.valid_dir)
    pred_x_json = "{}/pdtb-parses.json".format(args.pred_dir)
    output_json = "{}/output.json".format(args.output_dir)

    f = open(train_y_json, 'r')
    train_y_data = {}
    for line in f:
        relation = json.loads(line)

        # fix inconsistent structure
        if 'TokenList' not in relation['Arg1']:
            relation['Arg1']['TokenList'] = []
        if 'TokenList' not in relation['Arg2']:
            relation['Arg2']['TokenList'] = []
        if 'TokenList' not in relation['Connective']:
            relation['Connective']['TokenList'] = []

        # store by document id
        relation_len = len(relation['Arg1']['TokenList']) + len(relation['Arg2']['TokenList']) + len(relation['Connective']['TokenList'])
        try:
            train_y_data[relation['DocID']].append((relation_len, relation))
        except KeyError:
            train_y_data[relation['DocID']] = [(relation_len, relation)]

    # order by increasing discourse relation size
    all_tsenses = {}
    j = 0
    tag_to_j = {}
    j_to_tag = {}
    for doc_id in train_y_data:
        train_y_data[doc_id].sort()
        train_y_data[doc_id] = [ r  for _, r in train_y_data[doc_id] ][0:3]  #XXX only two relations!!!

        doc_tsenses = {}
        for relation in train_y_data[doc_id]:
            # count all senses and enumerate all tags
            for sense in relation['Sense']:
                tsense = "{0}:{1}".format(relation['Type'], sense)
                try:
                    all_tsenses[tsense] += 1
                except KeyError:
                    all_tsenses[tsense] = 1
                try:
                    doc_tsenses[tsense] += 1
                except KeyError:
                    doc_tsenses[tsense] = 1

                for part in ['Arg1', 'Arg2', 'Connective']:
                    if relation['Type'] == 'Implicit' and part == 'Connective':
                        continue

                    tag = "{0}:{1}:{2}:{3}".format(relation['Type'], sense, doc_tsenses[tsense] - 1, part)
                    if tag not in tag_to_j:
                        tag_to_j[tag] = j
                        j_to_tag[j] = tag
                        j += 1
    f.close()

    #doc_len_max = 0
    #train_x_it = data_pdtb.PDTBParsesCorpus(args.train_dir, with_document=True, with_paragraph=False, with_sentence=False, word_split="-|\\\\/", word_meta=True)
    #for doc in train_x_it:
    #    if len(doc) > doc_len_max:
    #        doc_len_max = len(doc)

    #import copy
    #train_y_out = copy.deepcopy(train_y_data[doc_id])
    #for relation in train_y_out:
    #    relation['Arg1']['TokenList'] = [ t[2]  for t in relation['Arg1']['TokenList'] ]
    #    relation['Arg2']['TokenList'] = [ t[2]  for t in relation['Arg2']['TokenList'] ]
    #    relation['Connective']['TokenList'] = [ t[2]  for t in relation['Connective']['TokenList'] ]

    #for relation in train_y_data[doc_id]:
    #    print relation['Type'], relation['Sense'][0], len(relation['Arg1']['TokenList']), len(relation['Arg2']['TokenList']), len(relation['Connective']['TokenList'])

    def train_x_infinite():
        while True:
            train_x_it = data_pdtb.PDTBParsesCorpus(args.train_dir, with_document=True, with_paragraph=False, with_sentence=False, word_split="-|\\\\/", word_meta=True)
            for doc in train_x_it:
                yield doc

    # settings
    learn_rate = 0.1
    decay_after = 10
    decay_rate = 0.95
    decay_min = 1e-8
    epochs = 10000
    rand_seed = int(time.time())
    x_dim = 300
    hidden_dim = 60 #XXX: x_dim
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
    for doc in train_x_infinite():
        doc_id = doc[0]['DocID']

        # map token ids to words
        map_i2words = {}
        for word in doc:
            for i in word['TokenList']:
                try:
                    map_i2words[i].append(word)
                except KeyError:
                    map_i2words[i] = [word]
                word['Tags'] = {}

        # put relation tags for each word
        doc_tsenses = {}
        for relation in train_y_data[doc_id]:
            for sense in relation['Sense']:
                tsense = "{0}:{1}".format(relation['Type'], sense)
                try:
                    doc_tsenses[tsense] += 1
                except KeyError:
                    doc_tsenses[tsense] = 1

                for part in ['Arg1', 'Arg2', 'Connective']:
                    for token_list in relation[part]['TokenList']:
                        for word in map_i2words[token_list[2]]:
                            tags = word['Tags']

                            tag = "{0}:{1}:{2}:{3}".format(relation['Type'], sense, doc_tsenses[tsense] - 1, part)
                            try:
                                tags[tag] += 1
                            except KeyError:
                                tags[tag] = 1

        # prepare training data
        train_x = []
        train_y = []
        for word in doc:
            # map text to word2vec
            try:
                train_x.append(map_word2vec[word['Text']])
            except KeyError:  # missing in vocab
                train_x.append([0.0] * x_dim)

            # map tags to vector
            tags = [0.0] * y_dim
            for tag, count in word['Tags'].iteritems():
                tags[tag_to_j[tag]] = float(count)
            train_y.append(tags)

            #print word['Text'], word['Tags']
            #print word['Text'], train_x[-1][0:1], train_y[-1]

        # train model
        t = time.time()
        cost, y_min, y_max, y_mean, y = rnn.train(train_x, train_y, np.array(learn_rate, dtype=np.float32))
        print "learning epoch {0} ({1:.2f} sec), rate {2:.2e}, train cost {3}{4}".format(epoch, time.time() - t, learn_rate, cost, (" +" if cost < best_cost else ""))
        print y_min, y_max, y_mean
        if cost < best_cost:
            best_cost = cost
            best_epoch = epoch

        # extract relations
        #y = rnn.predict(train_x)

        y_relations = []
        for relation in extract_relations(doc_id, doc, y, tag_to_j, all_tsenses):
            y_relations.append(relation)

            # check relations
            validator.check_type(relation)    
            validator.check_sense(relation)
            validator.check_args(relation)
            validator.check_connective(relation)

        # evaluate relations
        try:
            precision, recall, f1 = scorer.evaluate_relation(train_y_data[doc_id], y_relations)
            print "evaluate train set precision {0:.2f}, recall {1:.2f}, f1 {2:.2f}".format(precision, recall, f1)
        except ZeroDivisionError:
            precision, recall, f1 = 0.0, 0.0, 0.0
        if f1 == 1.0:
            print "WOOHOO!!!"
            break

        # learning rate decay if no improvement after some epochs
        if epoch - best_epoch >= decay_after:
            learn_rate *= decay_rate
        if epoch > epochs or learn_rate < decay_min:
            break
        epoch += 1
