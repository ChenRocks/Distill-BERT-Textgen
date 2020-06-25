"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

data for C-MLM
"""
import math
import random
import shelve
import warnings

import torch
from torch.utils.data import Dataset, Sampler

from toolz.sandbox.core import unzip


EOS = '</s>'
IN_WORD = '@@'
UNK = '<unk>'
UNK_BERT = '[UNK]'
MASK = '[MASK]'
CLS = '[CLS]'
SEP = '[SEP]'
MOSES_SPECIALS = {'&amp;': '&', '&#124;': '|', '&lt;': '<', '&gt;': '>',
                  '&apos;': "'", '&quot;': '"', '&#91;': '[', '&#93;': ']',
                  '@-@': '-'}


class BertDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, vocab, seq_len, max_len=150):
        self.db = shelve.open(corpus_path, 'r')
        self.lens = []
        self.ids = []
        for i, example in self.db.items():
            src_len = len(example['src'])
            tgt_len = len(example['tgt'])
            if (src_len <= max_len and tgt_len <= max_len):
                self.ids.append(i)
                self.lens.append(min(seq_len, src_len+tgt_len+3))
        self.vocab = vocab  # vocab for output seq2seq
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        item = self.db[id_]
        t1, t2 = item['src'], item['tgt']

        # combine to one sample
        cur_example = InputExample(guid=i, tokens_a=t1, tokens_b=t2)

        # transform sample to features
        cur_features = convert_example_to_features(
            cur_example, self.seq_len, self.tokenizer, self.vocab)

        features = (cur_features.input_ids, cur_features.input_mask,
                    cur_features.segment_ids, cur_features.lm_label_ids)

        return features

    @staticmethod
    def pad_collate(features):
        """ pad the input features to same length"""
        input_ids, input_masks, segment_ids, lm_label_ids = map(
            list, unzip(features))
        max_len = max(map(len, input_ids))
        for ids, masks, segs, labels in zip(input_ids, input_masks,
                                            segment_ids, lm_label_ids):
            while len(ids) < max_len:
                ids.append(0)
                masks.append(0)
                segs.append(0)
                labels.append(-1)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_masks)
        segment_ids = torch.tensor(segment_ids)
        lm_label_ids = torch.tensor(lm_label_ids)
        return input_ids, input_mask, segment_ids, lm_label_ids


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens_a, tokens_b=None, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For
                single sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.lm_labels = lm_labels  # masked words for language model


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask,
                 segment_ids, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then
    # each token that's truncated likely contains more information than a
    # longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_token_to_bert(token):
    bert_token = token.replace(IN_WORD, '')
    try:
        bert_token = MOSES_SPECIALS[bert_token]  # handle moses tokens
    except KeyError:
        pass
    if bert_token == UNK:
        # this should only happen with gigaword
        bert_token = UNK_BERT
    return bert_token


def convert_raw_input_to_features(src_line, tgt_line,
                                  toker, vocab, device, p=0.15):
    src_toks = [convert_token_to_bert(tok)
                for tok in src_line.strip().split()]
    tgt_toks = tgt_line.strip().split()
    output_labels = []
    for i, tok in enumerate(tgt_toks):
        if random.random() < p:
            tgt_toks[i] = MASK
            output_labels.append(vocab[tok])
        else:
            tgt_toks[i] = convert_token_to_bert(tok)
            output_labels.append(-1)
    if random.random() < p:
        tgt_toks.append(MASK)
        output_labels.append(vocab[EOS])
    else:
        tgt_toks.append(SEP)
        output_labels.append(-1)
    input_ids = toker.convert_tokens_to_ids(
        [CLS] + src_toks + [SEP] + tgt_toks)
    type_ids = [0]*(len(src_toks) + 2) + [1]*(len(tgt_toks))
    mask = [1] * len(input_ids)
    labels = [-1] * (len(src_toks) + 2) + output_labels
    input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)
    type_ids = torch.LongTensor(type_ids).to(device).unsqueeze(0)
    mask = torch.LongTensor(mask).to(device).unsqueeze(0)
    labels = torch.LongTensor(labels).to(device).unsqueeze(0)
    return input_ids, type_ids, mask, labels


def random_word(tokens, output_vocab):
    """
    NOTE: this assumes other MT prepro like moses and we try to align
          them with BERT
    Masking some random tokens for Language Model task with probabilities as in
    the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param output_vocab: vocab for seq2seq output
    :return: (list of str, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        # mask token with 15% probability
        if random.random() < 0.15:
            # we always MASK given our purpose
            tokens[i] = MASK

            # append current token to output (we will predict these later)
            try:
                output_label.append(output_vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(output_vocab[UNK])
                warnings.warn(f"Cannot find token '{token}' in vocab. Using "
                              f"{UNK} insetad")
        else:
            # handle input for BERT
            tokens[i] = convert_token_to_bert(token)

            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    # last SEP is used to learn EOS
    if random.random() < 0.15:
        tokens.append(MASK)
        output_label.append(output_vocab[EOS])
    else:
        tokens.append(SEP)
        output_label.append(-1)

    return tokens, output_label


def convert_example_to_features(example, max_seq_length,
                                tokenizer, output_vocab):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper
    training sample with IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :param output_vocab: vocab for seq2seq output
    :return: InputFeatures, containing all inputs and labels of one sample as
        IDs (as used for model training)
    """
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], EOS with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    # convert to BERT compatible inputs
    tokens_a = [convert_token_to_bert(tok) for tok in tokens_a]
    # only mask sent_b because it's seq2seq problem
    t1_label = [-1] * len(tokens_a)
    while True:
        tokens_b, t2_label = random_word(tokens_b, output_vocab)
        if any(label != -1 for label in t2_label):
            break
    # concatenate lm labels and account for CLS, SEP
    lm_label_ids = ([-1] + t1_label + [-1] + t2_label)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # For our MT setup
    # (b) For single sequences:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [MASK]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    tokens = []
    segment_ids = []
    tokens.append(CLS)
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append(SEP)
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    # NOTE the last SEP is handled differently from original BERT

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to multiples of 8 (for tensor cores)
    while len(input_ids) % 8 != 0:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) % 8 == 0
    assert (len(input_ids) == len(input_mask)
            == len(segment_ids) == len(lm_label_ids))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids)
    return features


class BucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size, droplast=False):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class DistributedBucketSampler(BucketSampler):
    def __init__(self, num_replicas, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._num_replicas = num_replicas

    def _create_ids(self):
        return super()._create_ids()[self._rank:-1:self._num_replicas]

    def __len__(self):
        num_data = len(self._create_ids())
        bucket_sizes = ([self._bucket_size]
                        * (num_data // self._bucket_size)
                        + [num_data % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)


class TokenBucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size, droplast=False):
        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for index in bucket:
                max_len = max(max_len, self._lens[index])
                if max_len * (len(batch_indices) + 1) > self._max_tok:
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    batches.append(batch_indices)
                    batch_indices = [index]
                else:
                    batch_indices.append(index)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class DistributedTokenBucketSampler(TokenBucketSampler):
    def __init__(self, num_replicas, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._num_replicas = num_replicas

    def _create_ids(self):
        return super()._create_ids()[self._rank:-1:self._num_replicas]
