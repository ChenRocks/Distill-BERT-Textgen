"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

detokenize output due to BERT preprocessing
"""
import argparse
import os
from os.path import basename, exists, join

import ipdb
from tqdm import tqdm

IN_WORD = '@@'
BERT_IN_WORD = '##'

# special chars in moses tokenizer
MOSES_SPECIALS = {'|': '&#124;', '<': '&lt;', '>': '&gt;',
                  "'": '&apos;', '"': '&quot;', '[': '&#91;', ']': '&#93;'}
AMP = '&'
AMP_MOSES = '&amp;'
UNK = '<unk>'


def convert_moses(tok):
    if tok in MOSES_SPECIALS:
        return MOSES_SPECIALS[tok]
    return tok


def detokenize(line, moses=True):
    word = ''
    words = []
    for tok in line.split():
        if tok.startswith(IN_WORD):
            tok = tok[2:]
            if tok.startswith(BERT_IN_WORD):
                tok = tok[2:]
            tok = tok.replace(AMP, AMP_MOSES)
            if moses:
                tok = convert_moses(tok)
            word += tok
        else:
            if tok.startswith(BERT_IN_WORD):
                ipdb.set_trace()
                raise ValueError()
            words.append(word)
            tok = tok.replace(AMP, AMP_MOSES)
            if moses:
                tok = convert_moses(tok)
            word = tok
    words.append(word)
    text = ' '.join(words).strip()
    return text


def process(reader, writer, unk, moses=True):
    for line in tqdm(reader, desc='tokenizing'):
        output = detokenize(line, moses)
        output = output.replace(UNK, unk)  # UNK format change
        writer.write(output + '\n')


def main(opts):
    if not exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    output_file = join(opts.output_dir, f'{basename(opts.file)}.detok')
    with open(opts.file, 'r') as reader, open(output_file, 'w') as writer:
        process(reader, writer, opts.unk, opts.moses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='store', required=True,
                        help='line by line text file for data')
    parser.add_argument('--output_dir', action='store', required=True,
                        help='path to output')
    parser.add_argument('--unk', action='store', default='UNK',
                        choices=['UNK', '<unk>'],
                        help='gigaword dev and test has different UNK')
    parser.add_argument('--no-moses', action='store_true',
                        help='turn off moses sepcial character mapping '
                             '(for gigaword)')
    args = parser.parse_args()
    args.moses = not args.no_moses
    main(args)
