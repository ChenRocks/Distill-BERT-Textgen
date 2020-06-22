FROM nvcr.io/nvidia/pytorch:19.03-py3

# python dependencies
RUN pip install \
    six==1.12.0 future==0.17.1 configargparse==0.14.0 tensorboardX==1.6 ipdb==0.12 \
    pytorch-pretrained-bert==0.6.1 tqdm==4.30.0 torchtext==0.4.0

# download pretrained BERT checkpoint
RUN python -c \
    "from pytorch_pretrained_bert import BertTokenizer, BertForPreTraining; m = 'bert-base-multilingual-cased'; BertTokenizer.from_pretrained(m); BertForPreTraining.from_pretrained(m)"

# moses for MT preprocessing
RUN git clone https://github.com/moses-smt/mosesdecoder.git /workspace/mosesdecoder && \
    cd /workspace/mosesdecoder && git checkout c054501 && rm -r .git && cd -

WORKDIR /src

# Summarization
#COPY ./scripts/setup_rouge.sh .
#RUN bash setup_rouge.sh && rm ./setup_rouge.sh
