#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
set -x
set -e
export PYTHONUNBUFFERED=1

MODEL=$1
CKPT=$2
SPLIT=$3    # dev/test
BEAM=$4     # beam size in beam search
ALPHA=$5    # length penalty

# German to English (IWSLT 15)
DATAPATH=/data/de-en
SRC=$DATAPATH/$SPLIT.de.bert
TGT=$DATAPATH/$SPLIT.en.bert
REF=$DATAPATH/ref/$SPLIT.en
TGT_LANG='en'

OUT_PATH=${MODEL}/output
EXP="ckpt_${CKPT}-beam_${BEAM}-alpha_${ALPHA}.${SPLIT}"
GPUID=0

echo "running IWSLT De-En translation with beam size ${BEAM}, length penalty ${ALPHA}"
if [ ! -d "$OUT_PATH" ]; then
    mkdir $OUT_PATH
fi

# run inference
python opennmt/translate.py -gpu ${GPUID} \
                            -model ${MODEL}/ckpt/model_step_${CKPT}.pt \
                            -src ${SRC} \
                            -tgt ${TGT} \
                            -output ${OUT_PATH}/${EXP}.${TGT_LANG} \
                            -log_file ${OUT_PATH}/${EXP}.log \
                            -beam_size ${BEAM} -alpha ${ALPHA} \
                            -length_penalty wu \
                            -replace_unk -verbose -fp32

# detokenize BERT BPE
python scripts/bert_detokenize.py --file ${OUT_PATH}/${EXP}.${TGT_LANG} \
                                  --output_dir ${OUT_PATH}

# evaluation
perl opennmt/tools/multi-bleu.perl $REF \
                                   < ${OUT_PATH}/${EXP}.${TGT_LANG}.detok \
                                   | tee ${OUT_PATH}/${EXP}.bleu
