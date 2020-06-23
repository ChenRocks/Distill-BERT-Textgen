"""
Further pre-compute the top-K prob to save memory
"""
import argparse
import io
import shelve

import numpy as np
import torch
from tqdm import tqdm


def tensor_loads(dump):
    with io.BytesIO(dump) as reader:
        obj = np.load(reader, allow_pickle=False)
        if isinstance(obj, np.ndarray):
            tensor = obj
        else:
            tensor = obj['arr_0']
    return tensor


def dump_topk(topk):
    logit, index = topk
    with io.BytesIO() as writer:
        torch.save((logit.cpu(), index.cpu()), writer)
        dump = writer.getvalue()
    return dump


def main(opts):
    linear = torch.load(f'{opts.bert_hidden}/linear.pt').cuda()
    with shelve.open(f'{opts.bert_hidden}/db', 'r') as db, \
         shelve.open(f'{opts.bert_hidden}/topk', 'c') as topk_db:
        for key, value in tqdm(db.items(),
                               total=len(db), desc='computing topk...'):
            bert_hidden = torch.tensor(tensor_loads(value)).cuda()
            topk = linear(bert_hidden).topk(dim=-1, k=opts.topk)
            dump = dump_topk(topk)
            topk_db[key] = dump


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_hidden', required=True,
                        help='path to saved bert hidden')
    parser.add_argument('--topk', type=int, default=128,
                        help='topk logits to pre-compute (can extract larger '
                             'K and then set to smaller value at training)')
    args = parser.parse_args()
    main(args)
