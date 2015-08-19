#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Neural dataflow machines
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import numpy as np
import theano
import theano.tensor as T

from keras import activations, initializations
from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix
from keras.layers.core import Layer


### Debugging

theano.config.floatX = 'float32'
theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'


### Classes

class Instruction(Layer):
    """Instruction block with 2 inputs and 1 output."""

    def __init__(self, tag_dim, strength_dim, value_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', inner_activation='hard_sigmoid', weights=None, truncate_gradient=-1, return_sequences=False):

        super(Instruction, self).__init__()
        self.tag_dim = tag_dim
        self.strength_dim = strength_dim
        self.value_dim = value_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        # input
        self.input = T.tensor3()

        # parameters of the model
        #XXX fixed tags
        self.arg1_tag = theano.shared(name="arg1_tag", value=np.array([0., 1.], dtype=theano.config.floatX))
        self.arg2_tag = theano.shared(name="arg2_tag", value=np.array([1., 0.], dtype=theano.config.floatX))
        self.out_tag = theano.shared(name="out_tag", value=np.array([1., 1.], dtype=theano.config.floatX))

        self.arg1_w = self.init((self.value_dim, self.value_dim))
        self.arg2_w = self.init((self.value_dim, self.value_dim))
        self.out_b = shared_zeros((self.value_dim))

        # bundle
        self.params = [
            self.arg1_tag, self.arg2_tag, self.out_tag,
            self.arg1_w, self.arg2_w, self.out_b,
        ]

        if weights is not None:
            self.set_weights(weights)

    def _step(self, in_tags, in_strengths, in_values, arg1_strength_tm1, cell1_strength_tm1, arg1_value_tm1, cell1_value_tm1, arg2_strength_tm1, cell2_strength_tm1, arg2_value_tm1, cell2_value_tm1):
        """One step of scan loop."""

        # average inputs weighted by normalized tag similarity and strength
        data1_strengths = in_strengths * T.sum(abs(in_tags - self.arg1_tag), 1)
        data2_strengths = in_strengths * T.sum(abs(in_tags - self.arg2_tag), 1)
        #XXX similarity != distance

        data1_strength = T.sum(data1_strengths)
        data2_strength = T.sum(data2_strengths)

        data1_value = in_values * data1_strengths.dimshuffle(0, 'x') / data1_strength
        data2_value = in_values * data2_strengths.dimshuffle(0, 'x') / data2_strength

        out_strength = arg1_strength_tm1 * arg2_strength_tm1

        # LSTM of input for each arg (input=data1_strength, forget=out_strength, output=1)
        def _simple_lstm(i_t, f_t, o_t, data, h_tm1, c_tm1):
            c_t = f_t * c_tm1 + i_t * data
            h_t = o_t * c_t
            return h_t, c_t

        arg1_strength, cell1_strength = _simple_lstm(data1_strength, out_strength, 1., data1_strength, arg1_strength_tm1, cell1_strength_tm1)
        arg1_value, cell1_value = _simple_lstm(data1_strength, out_strength, 1., data1_value, arg1_value_tm1, cell1_value_tm1)
        arg2_strength, cell2_strength = _simple_lstm(data2_strength, out_strength, 1., data2_strength, arg2_strength_tm1, cell2_strength_tm1)
        arg2_value, cell2_value = _simple_lstm(data2_strength, out_strength, 1., data2_value, arg2_value_tm1, cell2_value_tm1)

        # single layer feed-forward neural network
        #out_value = T.dot(self.arg1_w, arg1_value) + T.dot(self.arg2_w, arg2_value) + self.out_b
        out_value = data1_value

        return (self.out_tag, out_strength, out_value, arg1_strength, cell1_strength, arg1_value, cell1_value, arg2_strength, cell2_strength, arg2_value, cell2_value, 
            data1_strengths, data1_strength, data1_value)

#    def get_output(self, train):
#        """Compute the final output."""
#
#        # input convert from (examples, time, input_dim) to (time, examples, input_dim)
#        X = self.get_input(train) 
#        X = X.dimshuffle((1,0,2))
#
#        # computation
#        x_in = T.dot(X, self.w_in) + self.b_in
#
#        # scan loop
#        [outputs, memories], updates = theano.scan(
#            self._step, 
#            sequences=[xi, xf, xo, xc],
#            outputs_info=[
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
#                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
#            ], 
#            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c], 
#            truncate_gradient=self.truncate_gradient 
#        )
#
#        # output
#        if self.return_sequences:
#            return outputs.dimshuffle((1,0,2))
#        return outputs[-1]

    def get_config(self):
        return {
            'name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'init': self.init.__name__,
            'inner_init': self.inner_init.__name__,
            'activation': self.activation.__name__,
            'inner_activation': self.inner_activation.__name__,
            'truncate_gradient': self.truncate_gradient,
            'return_sequences': self.return_sequences,
        }




tag_dim = 2
strength_dim = 1
value_dim = 3

xin_strengths = np.array([
    1.,
    1.,
], dtype=theano.config.floatX)
xin_tags = np.array([
    [0., 1.],
    [1., 0.],
], dtype=theano.config.floatX)
xin_values = np.array([
    [0.1, 0.0, 0.2],
    [0.1, 0.5, 0.1],
], dtype=theano.config.floatX)
xout_value = [0.2, 0.5, 0.3]  #XXX

ins1 = Instruction(tag_dim=tag_dim, strength_dim=strength_dim, value_dim=value_dim, init='orthogonal', inner_init='orthogonal', activation='tanh', inner_activation='tanh')

in_tags = T.fmatrix()
in_strengths = T.fvector()
in_values = T.fmatrix()
arg1_strength_tm1 = alloc_zeros_matrix(strength_dim)
cell1_strength_tm1 = alloc_zeros_matrix(strength_dim)
arg1_value_tm1 = alloc_zeros_matrix(value_dim)
cell1_value_tm1 = alloc_zeros_matrix(value_dim)
arg2_strength_tm1 = alloc_zeros_matrix(strength_dim)
cell2_strength_tm1 = alloc_zeros_matrix(strength_dim)
arg2_value_tm1 = alloc_zeros_matrix(value_dim)
cell2_value_tm1 = alloc_zeros_matrix(value_dim)

fun1 = theano.function(inputs=[in_tags, in_strengths, in_values], outputs=ins1._step(in_tags, in_strengths, in_values, arg1_strength_tm1, cell1_strength_tm1, arg1_value_tm1, cell1_value_tm1, arg2_strength_tm1, cell2_strength_tm1, arg2_value_tm1, cell2_value_tm1))
#fun1 = theano.function(inputs=ins1.get_input(train=False), outputs=ins1.get_output(train=False))

print xout_value
for i, v in enumerate(fun1(xin_tags, xin_strengths, xin_values)):
    print i, v




def task_cumsum():
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM

    # train data (examples, time, data)
    X_train = [
        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [-1., -1.]],
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [-1., -1.]],
    ]
    Y_train = [
        [[-1., -1.], [0.1, 0.1], [0.2, 0.2], [0.3, 1.3]],
        [[-1., -1.], [0.1, 0.2], [0.4, 0.6], [0.9, 1.2]],
    ]
    ignored = 1

    model = Sequential()
    model.add(Instruction(init='orthogonal', inner_init='orthogonal', activation='tanh', inner_activation='tanh'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
    score = model.evaluate(X_train, Y_train, batch_size=16)
    print score
    for y in Y_train:
        print y
    for preds in model.predict(X_train):
        print preds
