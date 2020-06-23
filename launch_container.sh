# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA=$1
OUTPUT=$2

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

if [ ! -d $OUTPUT ]; then
    mkdir -p $OUTPUT
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/output,type=bind \
    --mount src=$DATA/dump,dst=/data,type=bind \
    chenrocks/distill-bert-textgen:latest
