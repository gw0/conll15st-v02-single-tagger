#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Neural dataflow machines
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import theano
import theano.tensor as T


def lstm(inputs, sig_remember, sig_forget, sig_output):
    pass  #TODO


def instruction_in2_out1(in_strengths, in_tags, in_values, arg1_tag, arg2_tag, out_tag, arg1_w, arg2_w, out_b):

    # average inputs weighted by normalized tag similarity and strength
	data1_strengths = in_strengths * T.sum(abs(in_tags - arg1_tag), 1)
	data2_strengths = in_strengths * T.sum(abs(in_tags - arg2_tag), 1)

	data1_strength_sum = T.sum(data1_strengths)
	data2_strength_sum = T.sum(data2_strengths)

	data1_value = in_values * data1_strengths.dimshuffle(0, 'x') / data1_strength_sum
    data2_value = in_values * data2_strengths.dimshuffle(0, 'x') / data2_strength_sum

    #TODO from future
	out_strength = arg1_strength * arg2_strength

    # LSTM of input for each arg
	arg1_strength, arg1_value = lstm([data1_strength, data1_value], data1_strength, out_strength, 1)
	arg2_strength, arg2_value = lstm([data2_strength, data2_value], data2_strength, out_strength, 1)

    # single layer feed-forward neural network
	out_value = T.dot(arg1_w, arg1_value) + T.dot(arg2_w, arg2_value) + out_b

	return (out_strength, out_tag, out_value)


tag_len = 2
value_len = 3

def x():
    in_strengths = T.fvector("in_strengths")
    in_tags = T.fmatrix("in_tags")
    in_values = T.fmatrix("in_values")

    #XXX fixed init
    arg1_tag = theano.shared(name="arg1_tag", value=np.array([0., 1.], dtype=theano.config.floatX))
    arg2_tag = theano.shared(name="arg2_tag", value=np.array([1., 0.], dtype=theano.config.floatX))
    out_tag = theano.shared(name="out_tag", value=np.array([1., 1.], dtype=theano.config.floatX))

    arg1_w = theano.shared(name="arg1_w", value=0.1 * np.random.uniform(-1., 1., (value_len, value_len)).astype(theano.config.floatX))
    arg2_w = theano.shared(name="arg2_w", value=0.1 * np.random.uniform(-1., 1., (value_len, value_len)).astype(theano.config.floatX))
    out_b = theano.shared(name="out_b", value=np.zeros(value_len, dtype=theano.config.floatX))

    ins1_strength, ins1_tag, ins1_value = instruction_in2_out1(in_strengths, in_tags, in_values, arg1_tag, arg2_tag, out_tag, arg1_w, arg2_w, out_b)

    return theano.function(inputs=[in_strengths, in_tags, in_values], outputs=[ins1_strength, ins1_tag, ins1_value])


in_strengths = [
    1.,
    1.,
]
in_tags = [
    [0., 1.],
    [1., 0.],
]
in_values = [
    [0.1, 0.0, 0.2],
    [0.1, 0.5, 0.1],
]
out_value = [0.2, 0.5, 0.3]  #XXX
fun1 = x()

print out_value
print fun1(in_strengths, in_tags, in_values)

