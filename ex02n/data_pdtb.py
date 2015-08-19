#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,C0326,W0621
"""
Parse PDTB dataset for CoNLL 2015 experiment 01 (Shallow Discourse Parsing).

Usage: ./data_pdtb.py <pdtb_dir>...

- <pdtb_dir>: conll15st dataset directory with PDTB parses and raw texts
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2015@tnode.com>"
__license__ = "GPLv3+"

import argparse
import json
import re

import common
log = common.logging.getLogger(__name__)


### Relation tags

tags_rnum1_most5 = [
    "EntRel:EntRel:1:Arg1",
    "EntRel:EntRel:1:Arg2",
    "EntRel:EntRel:1:Connective",
    "Explicit:Comparison.Contrast:1:Arg1",
    "Explicit:Comparison.Contrast:1:Arg2",
    "Explicit:Comparison.Contrast:1:Connective",
    "Explicit:Comparison:1:Arg1",
    "Explicit:Comparison:1:Arg2",
    "Explicit:Comparison:1:Connective",
    "Explicit:Contingency.Cause.Reason:1:Arg1",
    "Explicit:Contingency.Cause.Reason:1:Arg2",
    "Explicit:Contingency.Cause.Reason:1:Connective",
    "Explicit:Contingency.Condition:1:Arg1",
    "Explicit:Contingency.Condition:1:Arg2",
    "Explicit:Contingency.Condition:1:Connective",
    "Explicit:Expansion.Conjunction:1:Arg1",
    "Explicit:Expansion.Conjunction:1:Arg2",
    "Explicit:Expansion.Conjunction:1:Connective",
    "Explicit:Temporal.Asynchronous.Precedence:1:Arg1",
    "Explicit:Temporal.Asynchronous.Precedence:1:Arg2",
    "Explicit:Temporal.Asynchronous.Precedence:1:Connective",
    "Explicit:Temporal.Asynchronous.Succession:1:Arg1",
    "Explicit:Temporal.Asynchronous.Succession:1:Arg2",
    "Explicit:Temporal.Asynchronous.Succession:1:Connective",
    "Explicit:Temporal.Synchrony:1:Arg1",
    "Explicit:Temporal.Synchrony:1:Arg2",
    "Explicit:Temporal.Synchrony:1:Connective",
    "Implicit:Comparison.Contrast:1:Arg1",
    "Implicit:Comparison.Contrast:1:Arg2",
    "Implicit:Comparison.Contrast:1:Connective",
    "Implicit:Contingency.Cause.Reason:1:Arg1",
    "Implicit:Contingency.Cause.Reason:1:Arg2",
    "Implicit:Contingency.Cause.Reason:1:Connective",
    "Implicit:Contingency.Cause.Result:1:Arg1",
    "Implicit:Contingency.Cause.Result:1:Arg2",
    "Implicit:Contingency.Cause.Result:1:Connective",
    "Implicit:Expansion.Conjunction:1:Arg1",
    "Implicit:Expansion.Conjunction:1:Arg2",
    "Implicit:Expansion.Conjunction:1:Connective",
    "Implicit:Expansion.Instantiation:1:Arg1",
    "Implicit:Expansion.Instantiation:1:Arg2",
    "Implicit:Expansion.Instantiation:1:Connective",
    "Implicit:Expansion.Restatement:1:Arg1",
    "Implicit:Expansion.Restatement:1:Arg2",
    "Implicit:Expansion.Restatement:1:Connective",
    "Implicit:Temporal.Asynchronous.Precedence:1:Arg1",
    "Implicit:Temporal.Asynchronous.Precedence:1:Arg2",
    "Implicit:Temporal.Asynchronous.Precedence:1:Connective",
]


### PDTB parses

class PDTBParsesCorpus(object):
    """Iterate over tokens from the PDTB parses corpus at document, paragraph, sentence, or word level."""

    def __init__(self, pdtb_dirs, parses_ffmt="{}/pdtb-parses.json", raw_ffmt="{}/raw/{}", with_document=False, with_paragraph=False, with_sentence=False, paragraph_sep="^\W*\n\n\W*$", word_split=None, word_meta=False):
        if isinstance(pdtb_dirs, str):
            self.pdtb_dirs = [pdtb_dirs]
        else:
            self.pdtb_dirs = pdtb_dirs
        self.parses_ffmt = parses_ffmt
        self.raw_ffmt = raw_ffmt
        self.with_document = with_document  # include document level list
        self.with_paragraph = with_paragraph  # include paragraph level list
        self.with_sentence = with_sentence  # include sentence level list
        self.paragraph_sep = paragraph_sep  # regex to match paragraph separator
        self.word_split = word_split  # regex to split words
        self.word_meta = word_meta  # include word metadata

    def __iter__(self):

        def is_next_paragraph(fraw, prev_token_end, cur_token_begin):
            fraw.seek(prev_token_end)
            sep_str = fraw.read(cur_token_begin - prev_token_end)
            return re.match(self.paragraph_sep, sep_str, flags=re.MULTILINE)

        def split_token(token):
            if re.sub(self.word_split, "", token) == "":
                return [token]
            else:
                return re.split(self.word_split, token)

        for pdtb_dir in self.pdtb_dirs:
            f = open(self.parses_ffmt.format(pdtb_dir), 'r')

            for line in f:
                log.debug("- loading PDTB parses line (size {})".format(len(line)))
                parses_dict = json.loads(line)

                for doc_id in parses_dict:
                    document_level = []
                    paragraph_id = 0  # paragraph number within document
                    paragraph_level = []
                    sentence_id = 0  # sentence number within document
                    sentence_level = []
                    token_id = 0  # token number within document

                    fraw = open(self.raw_ffmt.format(pdtb_dir, doc_id), 'r')
                    prev_token_end = 0  # previous token last character offset

                    for sentence_dict in parses_dict[doc_id]['sentences']:
                        sentence_token_id = token_id  # first token number in sentence

                        for token in sentence_dict['words']:
                            if is_next_paragraph(fraw, prev_token_end, token[1]['CharacterOffsetBegin']):
                                if paragraph_level:
                                    if self.with_document:
                                        document_level.append(paragraph_level)
                                    else:
                                        yield paragraph_level
                                paragraph_id += 1
                                paragraph_level = []
                            prev_token_end = token[1]['CharacterOffsetEnd']

                            for word in split_token(token[0]):
                                if not self.word_meta:
                                    word_level = word
                                else:
                                    word_level = {
                                        'Text': word,
                                        'DocID': doc_id,
                                        'ParagraphID': paragraph_id,
                                        'SentenceID': sentence_id,
                                        'SentenceToken': sentence_token_id,
                                        'TokenList': [token_id],
                                        'PartOfSpeech': token[1]['PartOfSpeech'],
                                        'Linkers': token[1]['Linkers'],
                                    }

                                if self.with_sentence:
                                    sentence_level.append(word_level)
                                elif self.with_paragraph:
                                    paragraph_level.append(word_level)
                                elif self.with_document:
                                    document_level.append(word_level)
                                else:
                                    yield word_level
                            token_id += 1

                        if sentence_level:
                            if self.with_paragraph:
                                paragraph_level.append(sentence_level)
                            elif self.with_document:
                                document_level.append(sentence_level)
                            else:
                                yield sentence_level
                        sentence_id += 1
                        sentence_level = []

                    fraw.close()
                    if paragraph_level:
                        if self.with_document:
                            document_level.append(paragraph_level)
                        else:
                            yield paragraph_level
                    if document_level:
                        yield document_level

            f.close()


if __name__ == '__main__':
    # parse arguments
    argp = argparse.ArgumentParser(description="Parse PDTB dataset for CoNLL 2015 (Shallow Discourse Parsing).")
    argp.add_argument('pdtb_dir', nargs='+',
        help="conll15st dataset directory with PDTB parses and raw texts")
    args = argp.parse_args()

    log.info("Parsing {}...".format(args.pdtb_dir))
    it = PDTBParsesCorpus(args.pdtb_dir, with_document=False, with_paragraph=True, with_sentence=False, word_split="-|\\\\/", word_meta=False)
    for item in it:
        print item
