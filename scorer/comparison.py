#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0326,W0621
"""
Verify, score, and display performance of compared systems for conll15st.

Usage: ./comparison.py <gold_json> [-s <name> <json>]...

  - <gold_json>: gold standard to compare against in JSON format
  - <name>: system display name
  - <json>: system predicted output in JSON format

> ./comparison.py tutorial/pdtb_trial_data.json -s tutorial tutorial/pdtb_trial_system_output.json
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import joblib
import json
import tabulate
import numpy as np
import matplotlib.pyplot as plt
import itertools

import validator
import scorer

import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)


### Missing functions in scorer

def cm_avg_prf(cm):
    """Get average precision/recall/F1 scores for a confusion matrix."""

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    size = cm.alphabet.size()
    for i in xrange(size):
        label = cm.alphabet.get_label(i)
        if label == 'no':
            continue

        precision, recall, f1 = cm.get_prf(label)
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    size -= 1
    return (precision_sum / size, recall_sum / size, f1_sum / size)

def conv_gold_to_output(gold_list):
    """Convert in place gold standard data to output format."""

    for predict in gold_list:
        if isinstance(predict['Arg1']['TokenList'][0], list):
            predict['Arg1']['TokenList'] = [ off[2]  for off in predict['Arg1']['TokenList'] ]
        if isinstance(predict['Arg2']['TokenList'][0], list):
            predict['Arg2']['TokenList'] = [ off[2]  for off in predict['Arg2']['TokenList'] ]
        if 'TokenList' not in predict['Connective']:
            predict['Connective']['TokenList'] = []
        elif len(predict['Connective']['TokenList']) > 0 and isinstance(predict['Connective']['TokenList'][0], list):
            predict['Connective']['TokenList'] = [ off[2]  for off in predict['Connective']['TokenList'] ]
    return gold_list


### Scores

def scores_compute(gold_json, systems):
    """Verify and compute scores of all system outputs."""

    def to_percent(vals):
        return [ v * 100.0  for v in vals ]

    gold_list = [ json.loads(x) for x in open(gold_json) ]

    scores = {}
    for system_name, system_json in systems:
        log.debug("- validating system '{}' ('{}')...".format(system_name, system_json))
        if system_json != gold_json and not validator.validate_file(system_json):
            log.error("Invalid system output format in '{}' ('{}')!".format(system_name, system_json))
            exit(-1)

        log.debug("- scoring system '{}' ('{}')...".format(system_name, system_json))
        if system_json != gold_json:
            predicted_list = [ json.loads(x) for x in open(system_json) ]
        else:  # gold standard as system output
            import copy
            predicted_list = conv_gold_to_output(copy.deepcopy(gold_list))
        connective_cm, arg1_cm, arg2_cm, rel_arg_cm, sense_cm, precision, recall, f1 = scorer.evaluate(gold_list, predicted_list)

        scores[system_name] = {
            'conn': to_percent(connective_cm.get_prf('yes')),
            'arg1': to_percent(arg1_cm.get_prf('yes')),
            'arg2': to_percent(arg2_cm.get_prf('yes')),
            'comb': to_percent(rel_arg_cm.get_prf('yes')),
            'sense': to_percent(cm_avg_prf(sense_cm)),
            'overall': to_percent((precision, recall, f1)),
        }
    return scores

def scores_print(scores, system_names=None, subtask_names=None, transpose=False, tablefmt='simple', tuplefmt="{:4.1f}/{:4.1f}/{:4.1f}"):
    """Produce ASCII array of scores."""
    if system_names is None:
        system_names = sorted(scores.keys())
    if subtask_names is None:
        subtask_names = sorted(scores.itervalues().next().keys())

    if transpose == False:
        headers = ["method"] + subtask_names
        table = []
        for system_name in system_names:
            system_scores = [ tuplefmt.format(*scores[system_name][subtask_name])  for subtask_name in subtask_names ]
            table.append([system_name] + system_scores)

    else:
        headers = ["subtask"] + system_names
        table = []
        for subtask_name in subtask_names:
            subtask_scores = [ tuplefmt.format(*scores[system_name][subtask_name])  for system_name in system_names ]
            table.append([subtask_name] + subtask_scores)

    return tabulate.tabulate(table, headers=headers, tablefmt=tablefmt, stralign='right')

def scores_plot(fname, system_names, subtask_names, means_list, stds_list, title, xlabel, ylabel):
    """Produce plots of scores."""
    if isinstance(system_names[0], list):
        system_names = [ system_name  for system_name, _ in system_names ]
    if stds_list is None:
        stds_list = [ None  for _ in system_names ]
    color_list = list(itertools.islice(itertools.cycle(['b', 'g', 'r', 'c', 'm']), None, len(system_names)))

    offsets = np.arange(len(subtask_names))
    bar_width = 0.8 / len(system_names)
    opacity = 0.4

    fig, ax = plt.subplots()
    ax.set_aspect(6.0 / 100.0)

    for i in xrange(len(system_names)):
        plt.bar(offsets + i * bar_width, means_list[i], yerr=stds_list[i], error_kw={'ecolor': 0.3}, alpha=opacity, color=color_list[i], width=bar_width, label=system_names[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(offsets + bar_width, subtask_names)
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', dpi=100)

def scores_display(fname_pre, systems, scores):
    """Print and plot scores of all system outputs."""

    fname_latex = fname_pre + ".tex"
    fname_precision = fname_pre + "-precision.png"
    fname_recall = fname_pre + "-recall.png"
    fname_f1 = fname_pre + "-f1.png"

    # prepare data
    system_names = [ system_name  for system_name, _ in systems ]
    subtask_names = ['conn', 'arg1', 'arg2', 'comb', 'sense', 'overall']

    means_precision = []
    means_recall = []
    means_f1 = []
    for system_name in system_names:
        means_precision.append([ scores[system_name][subtask][0]  for subtask in subtask_names ])
        means_recall.append([ scores[system_name][subtask][1]  for subtask in subtask_names ])
        means_f1.append([ scores[system_name][subtask][2]  for subtask in subtask_names ])

    # print scores
    log.debug("- printing scores in LaTeX ('{}')...".format(fname_latex))
    f = open(fname_latex, 'w')
    f.write(scores_print(scores, system_names=system_names, subtask_names=subtask_names, transpose=True, tablefmt='latex'))
    f.close()

    log.debug("- printing scores in ASCII...")
    print(scores_print(scores, system_names=system_names, subtask_names=subtask_names, transpose=True, tablefmt='simple'))

    # plot scores
    log.info("- plotting precision ('{}')...".format(fname_precision))
    scores_plot(fname_precision, args.systems, subtask_names, means_precision, None, "", "subtasks by method", "precision")
    log.info("- plotting recall ('{}')...".format(fname_recall))
    scores_plot(fname_recall, args.systems, subtask_names, means_recall, None, "", "subtasks by method", "recall")
    log.info("- plotting F1-score ('{}')...".format(fname_f1))
    scores_plot(fname_f1, args.systems, subtask_names, means_f1, None, "", "subtasks by method", "F1-score")


if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description="Verify, score, and plot performances for conll15st.")
    argp.add_argument('gold_json',
        help="gold standard to compare against in JSON format")
    argp.add_argument("-s", dest='systems', action='append',
        nargs=2, metavar=("name", "json"),
        help="system display name and predicted output in JSON format")
    args = argp.parse_args()

    # compute scores
    log.info("Computing scores of system outputs...")
    scores = scores_compute(args.gold_json, args.systems)
    joblib.dump(scores, "scores.dump", compress=1)

    # print and plot scores
    log.debug("Displaying scores...")
    scores_display("./scores", args.systems, scores)

    #XXX
    log.info("Showing...")
    import os
    os.system("gwenview scores-precision.png 2> /dev/null &")
    os.system("gwenview scores-recall.png 2> /dev/null &")
    os.system("gwenview scores-f1.png 2> /dev/null &")
    #./comparison.py ../conll15-st-trial/pdtb-data.json -s baseline conll15-st-trial/output.json -s tutorial tutorial/pdtb_trial_system_output.json
