#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
set -x
set -e
export PYTHONUNBUFFERED=1

RAW=/data/raw
TMP=/data/tmp
DUMP=/data/dump
DOWNLOAD=/data/download

# download
echo "==========================================="
bash /src/scripts/download-iwslt_deen.sh $RAW $DOWNLOAD

RAW=$RAW/de-en
TMP=$TMP/de-en
DUMP=$DUMP/de-en


# BERT tokenization
python /src/scripts/bert_tokenize.py \
    --bert bert-base-multilingual-cased \
    --prefixes $RAW/train.en $RAW/train.de $RAW/valid $RAW/test \
    --output_dir $TMP


# prepare bert teacher training dataset
mkdir -p $DUMP
python /src/scripts/bert_prepro.py --src $TMP/train.de.bert \
                                   --tgt $TMP/train.en.bert \
                                   --output $DUMP/DEEN.db

# OpenNMT preprocessing
VSIZE=200000
FREQ=0
SHARD_SIZE=200000
python /src/opennmt/preprocess.py \
    -train_src $TMP/train.de.bert \
    -train_tgt $TMP/train.en.bert \
    -valid_src $TMP/valid.de.bert \
    -valid_tgt $TMP/valid.en.bert \
    -save_data $DUMP/DEEN \
    -src_seq_length 150 \
    -tgt_seq_length 150 \
    -src_vocab_size $VSIZE \
    -tgt_vocab_size $VSIZE \
    -vocab_size_multiple 8 \
    -src_words_min_frequency $FREQ \
    -tgt_words_min_frequency $FREQ \
    -share_vocab \
    -shard_size $SHARD_SIZE


# move needed files to dump
mv $TMP/valid.en.bert $DUMP/dev.en.bert
mv $TMP/valid.de.bert $DUMP/dev.de.bert
mv $TMP/test.en.bert $DUMP/test.en.bert
mv $TMP/test.de.bert $DUMP/test.de.bert
REFDIR=$DUMP/ref/
mkdir -p $REFDIR
cp $RAW/valid.en $REFDIR/dev.en
cp $RAW/test.en $REFDIR/test.en
