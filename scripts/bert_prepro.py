"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess text files for C-MLM finetuning
"""
import argparse
import shelve

from tqdm import tqdm


def make_db(src_reader, tgt_reader, db):
    print()
    for i, (src, tgt) in tqdm(enumerate(zip(src_reader, tgt_reader))):
        src_toks = src.strip().split()
        tgt_toks = tgt.strip().split()
        if src_toks and tgt_toks:
            dump = {'src': src_toks, 'tgt': tgt_toks}
        db[str(i)] = dump


def main(args):
    # process the dataset
    with open(args.src) as src_reader, open(args.tgt) as tgt_reader, \
         shelve.open(args.output, 'n') as db:
        make_db(src_reader, tgt_reader, db)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', action='store', required=True,
                        help='line by line text file for source data ')
    parser.add_argument('--tgt', action='store', required=True,
                        help='line by line text file for target data ')
    parser.add_argument('--output', action='store', required=True,
                        help='path to output')
    args = parser.parse_args()
    main(args)
