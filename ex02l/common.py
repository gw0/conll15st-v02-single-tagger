#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Common defaults and helpers for conll15st experiment 02.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import logging
import logging.config
import os
import time
import resource

import joblib


### Debug helpers

class Profiler(object):
    """Helper for monitoring time and memory usage."""

    def __init__(self, log):
        self.log = log
        self.start()

    def start(self):
        self.time_0 = time.time()
        self.mem_0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    def stop(self):
        self.time_1 = time.time()
        self.mem_1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.print_usage()

    def print_usage(self):
        self.log.error("(time {:.3f}s, memory {:+.1f}MB, total {:.3f}GB)".format(self.time_1 - self.time_0, (self.mem_1 - self.mem_0) / 1024.0, self.mem_1 / 1024.0 / 1024.0))


def profile(func, log=None):
    """Decorator for monitoring time and memory usage."""

    if log is None:
        log = logging.getLogger(func.__module__)
    profiler = Profiler(log)

    def wrap(*args, **kwargs):
        profiler.start()
        res = func(*args, **kwargs)
        profiler.stop()
        return res

    return wrap


### Defaults

log_modules = ['ex02']
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)

cache_dir = "./cache"
cache = joblib.Memory(cachedir=cache_dir, verbose=0).cache
