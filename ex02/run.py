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
import copy
from collections import defaultdict
from collections import OrderedDict

import common
import data_pdtb
import data_word2vec
from scorer import scorer, validator
log = common.logging.getLogger(__name__)

# debugging
theano.config.floatX = 'float32'
import socket
if socket.gethostname() == "elite":
    theano.config.optimizer = 'fast_compile'
    theano.config.exception_verbosity = 'high'


# similar to https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/rnnslu.py
class RNN_deep(object):

    def __init__(self, x_dim, hidden_dim, y_dim, w_spread, p_drop):

        # parameters of the model
        self.wx = theano.shared(name="wx", value=w_spread * np.random.uniform(-1.0, 1.0, (x_dim + hidden_dim + 1, hidden_dim)).astype(theano.config.floatX), borrow=True)
        self.hx_0 = theano.shared(name="hx_0", value=np.zeros(hidden_dim, dtype=theano.config.floatX), borrow=True)

        self.w1 = theano.shared(name="w1", value=w_spread * np.random.uniform(-1.0, 1.0, (hidden_dim + hidden_dim + 1, hidden_dim)).astype(theano.config.floatX), borrow=True)
        self.h1_0 = theano.shared(name="h1_0", value=np.zeros(hidden_dim, dtype=theano.config.floatX), borrow=True)

        self.wy = theano.shared(name="wy", value=w_spread * np.random.uniform(-1.0, 1.0, (hidden_dim + 1, y_dim)).astype(theano.config.floatX), borrow=True)

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

        #def sgd(cost, params, learn_rate):
        #    gradients = T.grad(cost, wrt=params)
        #    updates = [ (p, p - learn_rate * g)  for p, g in zip(params, gradients) ]
        #    return updates

        #def rmsprop(cost, params, learn_rate=0.01, rho=0.9, epsilon=1e-6):
        #    grads = T.grad(cost=cost, wrt=params)
        #    updates = []
        #    for p, g in zip(params, grads):
        #        acc = theano.shared(p.get_value() * 0.)
        #        acc_new = rho * acc + (1 - rho) * g ** 2
        #        gradient_scaling = T.sqrt(acc_new + epsilon)
        #        g = g / gradient_scaling
        #        updates.append((acc, acc_new))
        #        updates.append((p, p - learn_rate * g))
        #    return updates

        def adam(loss, all_params, learn_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
            """ADAM update rules
            
            Kingma, Diederik, and Jimmy Ba. "Adam: A Method for Stochastic Optimization." arXiv preprint arXiv:1412.6980 (2014). http://arxiv.org/pdf/1412.6980v4.pdf
            """
            updates = []
            all_grads = theano.grad(loss, all_params)
            alpha = learn_rate
            t = theano.shared(np.float32(1.0))
            b1_t = b1 * gamma ** (t - 1.0)   # decay the first moment running average coefficient
         
            for theta_prev, g in zip(all_params, all_grads):
                m_prev = theano.shared(np.zeros(theta_prev.get_value().shape, dtype=theano.config.floatX))
                v_prev = theano.shared(np.zeros(theta_prev.get_value().shape, dtype=theano.config.floatX))

                m = b1_t * m_prev + (1. - b1_t) * g  # update biased first moment estimate
                v = b2 * v_prev + (1. - b2) * g ** 2  # update biased second raw moment estimate
                m_hat = m / (1. - b1 ** t)  # compute bias-corrected first moment estimate
                v_hat = v / (1. - b2 ** t)  # compute bias-corrected second raw moment estimate
                theta = theta_prev - (alpha * m_hat) / (T.sqrt(v_hat) + e)  # update parameters

                updates.append((m_prev, m))
                updates.append((v_prev, v))
                updates.append((theta_prev, theta) )
            updates.append((t, t + 1.))
            return updates

        srng = T.shared_randomstreams.RandomStreams(np.random.randint(999999))
        def dropout_masks(p_drop, shapes):
            masks = [ srng.binomial(n=1, p=np.float32(1.) - p_drop, size=shape, dtype=theano.config.floatX)  for shape in shapes ]
            return masks
        def dropout_apply(h, mask, p_drop):
            if p_drop > 0.:
                h = h * mask / np.float32(1. - p_drop)
            return h

        def model(x, wx, hx_0, w1, h1_0, wy, p_drop):

            def recurrence(x_cur, hx_prev, h1_prev, masks):
                one = np.float32(1.)
                hx = activation(T.dot(T.concatenate([x_cur, hx_prev, [one]]), wx))
                hx_ = dropout_apply(hx, masks[0], p_drop)
                h1 = activation(T.dot(T.concatenate([hx_, h1_prev, [one]]), w1))
                h1_ = dropout_apply(h1, masks[1], p_drop)
                y_pred = activation(T.dot(T.concatenate([h1_, [one]]), wy))
                return (hx, h1, y_pred)

            if p_drop > 0.:
                masks = dropout_masks(theano.shared(np.float32(p_drop)), [ hx_0.shape, h1_0.shape ])
            else:
                masks = []
            (_, _, y_pred), _ = theano.scan(fn=recurrence, sequences=x, non_sequences=[masks], outputs_info=[hx_0, h1_0, None], n_steps=x.shape[0])
            return y_pred

        y_pred = model(x, self.wx, self.hx_0, self.w1, self.h1_0, self.wy, 0.)
        y_noise = model(x, self.wx, self.hx_0, self.w1, self.h1_0, self.wy, p_drop)

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
        cost = loss(y_noise, y) + l1_reg * l1 + l2_reg * l2
        #updates = sgd(cost, self.params, learn_rate)
        #updates = rmsprop(cost, self.params, learn_rate)
        updates = adam(cost, self.params, learn_rate)

        # compile theano functions
        self.predict = theano.function(inputs=[x], outputs=y_pred)
        self.train = theano.function(inputs=[x, y, learn_rate], outputs=[cost, T.min(y_noise), T.max(y_noise), T.mean(y_noise)], updates=updates)

    def save(self, dir):
        for param in self.params:
            joblib.dump(param.get_value(), "{}/{}.dump".format(dir, param.name), compress=0)

    def load(self, dir):
        for param in self.params:
            param.set_value(joblib.load("{}/{}.dump".format(dir, param.name)), borrow=True)


def relation_to_tag(relation, rpart):
    """Convert relation to tag."""

    rtype = relation['Type']
    rsense = relation['Sense'][0]  # assume only first sense
    rnum = relation['SenseNum'][0]  # assume only first sense
    return ":".join([rtype, rsense, rnum, rpart])


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
            relation['SenseNum'] = [str(rnums[rnum_key])]

            for rpart in ['Arg1', 'Arg2', 'Connective']:
                #if rnums[rnum_key] == 5:
                #    relation['SenseNum'] = [str(1)]
                #    print relation_to_tag(relation, rpart)
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


@common.cache
def load(pdtb_dir, word2vec_bin, word2vec_dim, tag_to_j):
    """Load PDTB data and transform it to numerical form."""

    parses_ffmt = "{}/pdtb-parses.json"
    relations_ffmt = "{}/pdtb-data.json"
    raw_ffmt = "{}/raw/{}"

    # load relations
    relations = load_relations(relations_ffmt.format(pdtb_dir), tag_to_j)

    # prepare mapping vocabulary to word2vec vectors
    map_word2vec = joblib.load("./ex02_model/map_word2vec.dump")  #XXX

    # load words from PDTB parses
    words = load_words(pdtb_dir, relations)

    # prepare numeric form
    x = []
    y = []
    doc_ids = []
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

        x.append(np.asarray(doc_x, dtype=theano.config.floatX))
        y.append(np.asarray(doc_y, dtype=theano.config.floatX))
        doc_ids.append(doc_id)
        if doc_id not in relations:
            relations[doc_id] = []

    return x, y, doc_ids, words, relations


def extract_relations(y, tag_to_j, words, thres=0.5):
    """Extract all relations above threshold into PDTB format."""

    doc_id = words[0]['DocID']

    # iterate through all tags
    relations_dict = {}
    for tag in tag_to_j:
        rtype, rsense, rnum, rpart = tag.split(":")
        relation_key = (rtype, rsense, rnum)

        if relation_key in relations_dict:  # previous relation
            relation = relations_dict[relation_key]
        else:  # new relation
            relation = {}
            relation['DocID'] = doc_id
            relation['Type'] = rtype
            relation['Sense'] = [rsense]
            relation['SenseNum'] = [rnum]
            relation['Arg1'] = {'TokenList': []}
            relation['Arg2'] = {'TokenList': []}
            relation['Connective'] = {'TokenList': []}
            relations_dict[relation_key] = relation

        # transform to token list
        j = tag_to_j[tag]
        token_list = {}
        for word, tags in zip(words, y):
            if tags[j] > thres:
                for k in word['TokenList']:
                    token_list[k] = 1
        relation[rpart]['TokenList'] = sorted(token_list.keys())

        if rtype == 'Implicit' and rpart == 'Connective':  # invalid case
            relation[rpart]['TokenList'] = []

    # extract meaningful relations
    relations = []
    for relation in relations_dict.itervalues():
        arg1_len = len(relation['Arg1']['TokenList'])
        arg2_len = len(relation['Arg2']['TokenList'])
        conn_len = len(relation['Connective']['TokenList'])
        if arg1_len > 2 and arg2_len > 2 and (arg1_len + arg2_len + conn_len) < 300:
            relations.append(relation)
        #if (arg1_len + arg2_len + conn_len) > 0:
        #    print ":".join([rtype, rsense, rnum]), arg1_len, arg2_len, conn_len

    # check relations
    #for relation in relations:
    #    validator.check_type(relation)    
    #    validator.check_sense(relation)
    #    validator.check_args(relation)
    #    validator.check_connective(relation)

    return relations


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
    log.info("load data")
    output_json = "{}/output.json".format(args.output_dir)
    word2vec_bin = "./GoogleNews-vectors-negative300.bin.gz"
    word2vec_dim = 300

    tag_to_j = {}
    #tag_to_j["Explicit:Expansion.Conjunction:1:Arg1"] = len(tag_to_j)
    #tag_to_j["Explicit:Expansion.Conjunction:1:Arg2"] = len(tag_to_j)
    #tag_to_j["Explicit:Expansion.Conjunction:1:Connective"] = len(tag_to_j)
    for i, tag in enumerate(data_pdtb.tags_rnum1_most5):
        tag_to_j[tag] = i

    x_train, y_train, train_doc_ids, train_words, train_relations = load(args.train_dir, word2vec_bin, word2vec_dim, tag_to_j)
    x_valid, y_valid, valid_doc_ids, valid_words, valid_relations = load(args.valid_dir, word2vec_bin, word2vec_dim, tag_to_j)
    x_test, _, test_doc_ids, test_words, _ = load(args.test_dir, word2vec_bin, word2vec_dim, tag_to_j)

    train_words_list = [ train_words[doc_id]  for doc_id in train_doc_ids ]
    train_relations_list = [ r  for doc_id in train_doc_ids for r in train_relations[doc_id] ]
    valid_words_list = [ valid_words[doc_id]  for doc_id in valid_doc_ids ]
    valid_relations_list = [ r  for doc_id in valid_doc_ids for r in valid_relations[doc_id] ]

    # test generate perfect output
    #import copy
    #train_relations_out = copy.deepcopy(train_relations["wsj_1000"])
    #for relation in train_relations_out:
    #    relation['Arg1']['TokenList'] = [ t[2]  for t in relation['Arg1']['TokenList'] ]
    #    relation['Arg2']['TokenList'] = [ t[2]  for t in relation['Arg2']['TokenList'] ]
    #    relation['Connective']['TokenList'] = [ t[2]  for t in relation['Connective']['TokenList'] ]

    # settings
    log.info("instantiate model")
    rand_seed = int(time.time())
    learn_rate = 0.01  # dataset trial=0.01, dev=0.001
    decay_after = 10
    decay_rate = 0.9
    decay_min = learn_rate * 1e-6
    epochs = 10000
    x_dim = word2vec_dim
    hidden_dim = 60  #XXX: x_dim
    y_dim = len(tag_to_j)
    w_spread = 0.1  # dim 30=0.1, 300=0.05
    p_drop = 0.
    valid_freq = 1

    # instantiate the model
    np.random.seed(rand_seed)
    rnn = RNN_deep(x_dim=x_dim, hidden_dim=hidden_dim, y_dim=y_dim, w_spread=w_spread, p_drop=p_drop)
    #rnn.load(args.model_dir)

    # iterate through train dataset
    log.info("learning and evaluating")
    best_train_cost = np.inf
    best_train_epoch = 0
    best_f1 = -np.inf
    best_epoch = 0
    #best_rnn = rnn
    epoch = 0
    while epoch < epochs or learn_rate > decay_min:

        # train model
        t = time.time()
        cost_avg = 0.0
        cost_min = np.inf
        cost_max = -np.inf
        y_min_avg = y_max_avg = y_mean_avg = 0.0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            cost, y_min, y_max, y_mean = rnn.train(x, y, np.float32(learn_rate))
            cost_avg += cost
            if cost < cost_min:
                cost_min = cost
            if cost > cost_max:
                cost_max = cost
            y_min_avg += y_min
            y_max_avg += y_max
            y_mean_avg += y_mean

            if i % int(len(x_train) / 4 + 1) == 0 and time.time() - t > 10:
                log.debug("learning epoch {} ({:.2f}%)".format(epoch, (i + 1) * 100.0 / len(x_train)))
        cost_avg /= len(x_train)
        y_min_avg /= len(x_train)
        y_max_avg /= len(x_train)
        y_mean_avg /= len(x_train)
        log.info("learning epoch {} ({:.2f} sec), rate {:.2e}, train cost ({:.4f} {:.4f}) avg {}{}".format(epoch, time.time() - t, learn_rate, float(cost_min), float(cost_max), cost_avg, (" +" if cost_avg < best_train_cost else "  ")))
        log.debug("  {} {} {}".format(y_min_avg, y_max_avg, y_mean_avg))
        if cost_avg < best_train_cost:
            best_train_cost = cost_avg
            best_train_epoch = epoch

        # validate model
        if epoch % valid_freq == 0:
            t = time.time()
            y_relations = []
            for i, (x, y, words) in enumerate(zip(x_valid, y_valid, valid_words_list)):
                y_pred = rnn.predict(x)

                # extract relations from current document
                y_relations.extend(extract_relations(y_pred, tag_to_j, words))

            # evaluate all relations
            precision, recall, f1 = scorer.evaluate_relation(valid_relations_list, y_relations)
            log.info("valid set ({:.2f} sec): precision {:.2f}, recall {:.2f}, f1 {:.2f}".format(time.time() - t, precision, recall, f1))
            if f1 > best_f1:  # save best model
                best_f1 = f1
                best_epoch = epoch
                #best_rnn = copy.deepcopy(rnn)
                rnn.save(args.model_dir)
            if f1 >= 1.0:  # perfect
                print "  WOOHOO!!!"
                break

        # learning rate decay if no improvement after some epochs
        if epoch - best_train_epoch >= decay_after:
            learn_rate *= decay_rate
            #rnn = best_rnn
        epoch += 1
