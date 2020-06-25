# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# modified from hugginface github

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc.
# team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""C-MLM finetuning runner."""
import argparse
import copy
import json
import logging
import os
from os.path import abspath, dirname, exists, join
import random
import subprocess
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from cmlm.data import (BertDataset, TokenBucketSampler,
                       DistributedTokenBucketSampler,
                       convert_raw_input_to_features)
from cmlm.model import convert_embedding, BertForSeq2seq
from cmlm.util import Logger, RunningMeter
from cmlm.distributed import broadcast_tensors

# add opennmt to python module search path
# other than distributed utils, this is also needed to load onmt vocab file
import sys
sys.path.insert(0, '/src/opennmt')
from onmt.utils.distributed import (all_reduce_and_rescale_tensors,
                                    all_gather_list)


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

TB_LOGGER = Logger()


def noam_schedule(step, warmup_step=4000):
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def main(opts):
    if opts.local_rank == -1:
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        n_gpu = 1
    else:
        torch.cuda.set_device(opts.local_rank)
        device = torch.device("cuda", opts.local_rank)
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = torch.distributed.get_world_size()
        logger.info("device: {} n_gpu: {}, distributed training: {}, "
                    "16-bits training: {}".format(
                        device, n_gpu, bool(opts.local_rank != -1), opts.fp16))
    opts.n_gpu = n_gpu

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    is_master = opts.local_rank == -1 or torch.distributed.get_rank() == 0

    if is_master:
        save_training_meta(opts)

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opts.seed)

    tokenizer = BertTokenizer.from_pretrained(
        opts.bert_model, do_lower_case='uncased' in opts.bert_model)

    # train_examples = None
    print("Loading Train Dataset", opts.train_file)
    vocab_dump = torch.load(opts.vocab_file)
    vocab = vocab_dump['tgt'].fields[0][1].vocab.stoi
    train_dataset = BertDataset(opts.train_file, tokenizer, vocab,
                                seq_len=opts.max_seq_length,
                                max_len=opts.max_sent_length)

    # Prepare model
    model = BertForSeq2seq.from_pretrained(opts.bert_model)
    embedding = convert_embedding(
        tokenizer, vocab, model.bert.embeddings.word_embeddings.weight)
    model.update_output_layer(embedding)
    if opts.fp16:
        model.half()
    model.to(device)
    if opts.local_rank != -1:
        # need to make sure models are the same in the beginning
        params = [p.data for p in model.parameters()]
        broadcast_tensors(params)
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            module.p = opts.dropout

    # Prepare optimizer
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'pooler' not in n]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    if opts.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex "
                              "to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=opts.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if opts.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=opts.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=opts.learning_rate,
                             warmup=opts.warmup_proportion,
                             t_total=opts.num_train_steps)

    global_step = 0
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", opts.train_batch_size)
    logger.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    logger.info("  Num steps = %d", opts.num_train_steps)

    if opts.local_rank == -1:
        train_sampler = TokenBucketSampler(
            train_dataset.lens,
            bucket_size=8192,
            batch_size=opts.train_batch_size,
            droplast=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_sampler=train_sampler,
                                      num_workers=4,
                                      collate_fn=BertDataset.pad_collate)
    else:
        train_sampler = DistributedTokenBucketSampler(
            n_gpu, opts.local_rank,
            train_dataset.lens,
            bucket_size=8192,
            batch_size=opts.train_batch_size,
            droplast=True)
        train_dataloader = DataLoader(train_dataset,
                                      batch_sampler=train_sampler,
                                      num_workers=4,
                                      collate_fn=BertDataset.pad_collate)

    if is_master:
        TB_LOGGER.create(join(opts.output_dir, 'log'))
    running_loss = RunningMeter('loss')
    model.train()
    if is_master:
        pbar = tqdm(total=opts.num_train_steps)
    else:
        logger.disabled = True
        pbar = None
    n_examples = 0
    n_epoch = 0
    start = time()
    while True:
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) if t is not None else t for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            n_examples += input_ids.size(0)
            mask = lm_label_ids != -1
            loss = model(input_ids, segment_ids, input_mask,
                         lm_label_ids, mask, True)
            if opts.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            running_loss(loss.item())
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1
                if opts.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if opts.fp16 is False, BertAdam is used that handles
                    # this automatically
                    lr_this_step = opts.learning_rate * warmup_linear(
                        global_step/opts.num_train_steps,
                        opts.warmup_proportion)
                    if lr_this_step < 0:
                        # save guard for possible miscalculation of train steps
                        lr_this_step = 1e-8
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    TB_LOGGER.add_scalar('lr',
                                         lr_this_step, global_step)

                # NOTE running loss not gathered across GPUs for speed
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                if opts.local_rank != -1:
                    # gather gradients from every processes
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))
                optimizer.step()
                optimizer.zero_grad()
                if pbar is not None:
                    pbar.update(1)
                if global_step % 5 == 0:
                    torch.cuda.empty_cache()
                if global_step % 100 == 0:
                    if opts.local_rank != -1:
                        total = sum(all_gather_list(n_examples))
                    else:
                        total = n_examples
                    if is_master:
                        ex_per_sec = int(total / (time()-start))
                        logger.info(f'{total} examples trained at '
                                    f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('ex_per_s', ex_per_sec, global_step)

                if global_step % opts.valid_steps == 0:
                    logger.info(f"start validation at Step {global_step}")
                    with torch.no_grad():
                        val_log = validate(model,
                                           opts.valid_src, opts.valid_tgt,
                                           tokenizer, vocab, device,
                                           opts.local_rank)
                    logger.info(f"Val Acc: {val_log['val_acc']}; "
                                f"Val Loss: {val_log['val_loss']}")
                    TB_LOGGER.log_scaler_dict(val_log)
                    if is_master:
                        output_model_file = join(
                            opts.output_dir, 'ckpt',
                            f"model_step_{global_step}.pt")
                        # save cpu checkpoint
                        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor)
                                      else v
                                      for k, v in model.state_dict().items()}
                        torch.save(state_dict, output_model_file)
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        if is_master:
            logger.info(f"finished {n_epoch} epochs")
    if opts.num_train_steps % opts.valid_steps != 0:
        with torch.no_grad():
            val_log = validate(model, opts.valid_src, opts.valid_tgt,
                               tokenizer, vocab, device, opts.local_rank)
        TB_LOGGER.log_scaler_dict(val_log)
        if is_master:
            output_model_file = join(opts.output_dir, 'ckpt',
                                     f"model_step_{global_step}.pt")
            # save cpu checkpoint
            state_dict = {k: v.cpu() if isinstance(v, torch.Tensor)
                          else v
                          for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), output_model_file)


def validate(model, valid_src, valid_tgt, toker, vocab, device, local_rank):
    model.eval()
    val_loss = 0
    n_correct = 0
    n_word = 0
    with open(valid_src, 'r') as src_reader, \
         open(valid_tgt, 'r') as tgt_reader:
        for i, (src, tgt) in enumerate(zip(src_reader, tgt_reader)):
            if local_rank != -1:
                global_rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                if global_rank % world_size != 0:
                    continue
            input_ids, type_ids, mask, labels = convert_raw_input_to_features(
                src, tgt, toker, vocab, device)
            prediction_scores = model(input_ids, type_ids, mask)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1,
                                                 reduction='sum')
            loss = loss_fct(prediction_scores.squeeze(0), labels.view(-1))
            val_loss += loss.item()
            n_correct += accuracy_count(prediction_scores, labels)
            n_word += (labels != -1).long().sum().item()
    if local_rank != -1:
        val_loss = sum(all_gather_list(val_loss))
        n_correct = sum(all_gather_list(n_correct))
        n_word = sum(all_gather_list(n_word))
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'val_loss': val_loss, 'val_acc': acc}
    model.train()
    return val_log


def accuracy_count(out, labels):
    outputs = out.max(dim=-1)[1]
    mask = labels != -1
    n_correct = (outputs == labels).masked_select(mask).sum().item()
    return n_correct


def save_training_meta(opts):
    if not exists(opts.output_dir):
        os.makedirs(join(opts.output_dir, 'log'))
        os.makedirs(join(opts.output_dir, 'ckpt'))

    with open(join(opts.output_dir, 'log', 'hps.json'), 'w') as writer:
        hps = copy.deepcopy(vars(opts))
        del hps['local_rank']
        json.dump(hps, writer, indent=4)
    # git info
    try:
        logger.info("Waiting on git info....")
        c = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_branch_name = c.stdout.decode().strip()
        logger.info("Git branch: %s", git_branch_name)
        c = subprocess.run(["git", "rev-parse", "HEAD"],
                           timeout=10, stdout=subprocess.PIPE)
        git_sha = c.stdout.decode().strip()
        logger.info("Git SHA: %s", git_sha)
        git_dir = abspath(dirname(__file__))
        git_status = subprocess.check_output(
            ['git', 'status', '--short'],
            cwd=git_dir, universal_newlines=True).strip()
        with open(join(opts.output_dir, 'log', 'git_info.json'),
                  'w') as writer:
            json.dump({'branch': git_branch_name,
                       'is_dirty': bool(git_status),
                       'status': git_status,
                       'sha': git_sha},
                      writer, indent=4)
    except subprocess.TimeoutExpired as e:
        logger.exception(e)
        logger.warn("Git info not found. Moving right along...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The input train corpus. (shelve DB)")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="seq2seq output vocab")
    parser.add_argument("--valid_src", default=None, type=str, required=True,
                        help="source line txt for validation")
    parser.add_argument("--valid_tgt", default=None, type=str, required=True,
                        help="target line txt for validation")

    parser.add_argument(
        "--bert_model", default=None, type=str, required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, "
             "bert-base-chinese.")
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Other parameters
    parser.add_argument(
        "--max_seq_length", default=256, type=int,
        help="The maximum total input sequence length after WordPiece "
             "tokenization. \nSequences longer than this will be truncated, "
             "and sequences shorter \nthan this will be padded.")
    parser.add_argument("--max_sent_length", default=256, type=int,
                        help="The maximum number of tokens in a sentence")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear "
                             "learning rate warmup for. (linear decay)"
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. "
                             "Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling "
                             "value.\n")

    args = parser.parse_args()

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    main(args)
