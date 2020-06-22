# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DATA=$1

if [ ! -d $DATA ]; then
    mkdir -p $DATA
fi

CMD='./scripts/prepare-iwslt_deen.sh'

docker run --rm \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$DATA,dst=/data,type=bind \
    chenrocks/distill-bert-textgen:latest \
    bash -c $CMD
