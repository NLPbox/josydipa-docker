#!/usr/bin/env python3

"""This module converts .mrg files (syntax plus POS) from the Penn Treebank
into .cleangold format (as it is used by the josydipa RST parser.
"""

import argparse
import codecs
import os
import re
import sys

from nltk.corpus import BracketParseCorpusReader
from nltk.tree import Tree


# We want to keep only the chuck tag (without the relation tag), i.e. PP-CLR -> PP
CHUNK_TAG_RE = re.compile('^(?P<prefix>[A-Z]+)-(.*)$')


def read_mrg(mrg_filepath):
    """Parses a Penn Treebank .mrg (syntax+pos) file into an nltk Tree."""
    mrg_path, mrg_filename = os.path.split(mrg_filepath)
    return BracketParseCorpusReader(mrg_path, [mrg_filename]).parsed_sents()


def wrap_sentence(sent_tree, wrap_elem='TOP'):
    """Put an element on top of a nltk Tree."""
    return Tree(wrap_elem, [sent_tree])


def trim_labels(tree, trim_regex=CHUNK_TAG_RE):
    """Remove the relation tag from each non-terminal node.
    For example, 'PP-CLR' becomes 'PP'.
    """
    for pos in tree.treepositions():
        subtree = tree[pos]
        if isinstance(subtree, Tree):
            match = trim_regex.search(subtree.label())
            if not match:
                continue
            subtree.set_label(match.groupdict()['prefix'])
    return tree


def sentence_mrg2cleangold(sent_tree):
    """Converts an nltk Tree representation of a sentence in .mrg format
    into a string representation of a sentence in .cleangold format.
    """
    output_tree = wrap_sentence(trim_labels(sent_tree))
    return " {}".format(" ".join(output_tree.pformat().split()))


def mrg2cleangold(mrg_filepath, output_filepath=None):
    """Read a .mrg file, convert it to .cleangold format and write the output
    to the given file (or STDOUT).
    """
    sentences = read_mrg(mrg_filepath)
    if isinstance(output_filepath, str):
        with codecs.open(output_filepath, 'w', 'utf-8') as outfile:
            for sent in sentences:
                output_sent_str = sentence_mrg2cleangold(sent)
                outfile.write(output_sent_str + "\n")
    else:
        for sent in sentences:
            print(sentence_mrg2cleangold(sent))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mrg_file', default=sys.stdout,
                        help="The .mrg file to be parsed.")
    parser.add_argument('output_file', nargs='?', default=sys.stdout,
                        help="The file to write the .cleangold output to.")

    args = parser.parse_args(sys.argv[1:])
    mrg2cleangold(args.mrg_file, args.output_file)
