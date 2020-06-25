#!/usr/bin/env bash
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

set -x
set -e

RAW=$1
TMP=$2


SCRIPTS=/workspace/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

URL="https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz"
GZ="de-en.tgz"

src=de
tgt=en
lang=de-en

prep=$RAW/de-en
orig=$TMP


mkdir -p $orig $prep
cd $orig

if [ -f $GZ ]; then
    echo "$file already exists, skipping download"
else
    echo "Downloading data from ${URL}..."
    wget "$URL"
    if [ -f $GZ ]; then
        echo "Data successfully downloaded."
    else
        echo "Data not successfully downloaded."
        exit
    fi
    tar zxvf $GZ
fi
cd -

if [ -f $prep/train.en ] && [ -f $prep/train.de ] && \
    [ -f $prep/valid.en ] && [ -f $prep/valid.de ] && \
    [ -f $prep/test.en ] && [ -f $prep/test.de ]; then
    echo "iwslt dataset is already preprocessed, skip"
else
    echo "pre-processing train data..."
    for l in $src $tgt; do
        f=train.tags.$lang.$l
        tok=train.tags.$lang.tok.$l

        cat $orig/$lang/$f | \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        sed -e 's/<title>//g' | \
        sed -e 's/<\/title>//g' | \
        sed -e 's/<description>//g' | \
        sed -e 's/<\/description>//g' | \
        perl $TOKENIZER -threads 8 -l $l > $prep/$tok
        echo ""
    done
    perl $CLEAN -ratio 1.5 $prep/train.tags.$lang.tok $src $tgt $prep/train.tags.$lang.clean 1 175
    for l in $src $tgt; do
        perl $LC < $prep/train.tags.$lang.clean.$l > $prep/train.tags.$lang.$l
    done

    echo "pre-processing valid/test data..."
    for l in $src $tgt; do
        for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$prep/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -l $l | \
        perl $LC > $f
        echo ""
        done
    done


    echo "creating train, valid, test..."
    for l in $src $tgt; do
        awk '{if (NR%23 == 0)  print $0; }' $prep/train.tags.de-en.$l > $prep/valid.$l
        awk '{if (NR%23 != 0)  print $0; }' $prep/train.tags.de-en.$l > $prep/train.$l

        cat $prep/IWSLT14.TED.dev2010.de-en.$l \
            $prep/IWSLT14.TEDX.dev2012.de-en.$l \
            $prep/IWSLT14.TED.tst2010.de-en.$l \
            $prep/IWSLT14.TED.tst2011.de-en.$l \
            $prep/IWSLT14.TED.tst2012.de-en.$l \
            > $prep/test.$l
    done
fi
