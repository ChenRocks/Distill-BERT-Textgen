"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

C-MLM model
"""
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import BertForMaskedLM


IN_WORD = '@@'


def convert_embedding(toker, vocab, emb_weight):
    """ seq2seq vs pretrained BERT embedding conversion"""
    vocab_size = emb_weight.size(1)
    if vocab_size % 8 != 0:
        # pad for tensor cores
        vocab_size += (8 - vocab_size % 8)
    vectors = [torch.zeros(vocab_size) for _ in range(len(vocab))]
    for word, id_ in vocab.items():
        word = word.replace(IN_WORD, '')
        if word in toker.vocab:
            bert_id = toker.vocab[word]
        else:
            bert_id = toker.vocab['[UNK]']
        vectors[id_] = emb_weight[bert_id].clone()
    embedding = nn.Parameter(torch.stack(vectors, dim=0))
    return embedding


class BertForSeq2seq(BertForMaskedLM):
    """
    The original output projection is shared w/ embedding. Now for seq2seq, we
    use initilization from bert embedding but untied embedding due to
    tokenization difference
    """
    def __init__(self, config, causal=False):
        super().__init__(config)
        self.apply(self.init_bert_weights)

    def update_output_layer(self, output_embedding):
        self.cls.predictions.decoder.weight = output_embedding
        vocab_size = output_embedding.size(0)
        self.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size))
        self.config.vocab_size = vocab_size

    def update_output_layer_by_size(self, vocab_size):
        if vocab_size % 8 != 0:
            # pad for tensor cores
            vocab_size += (8 - vocab_size % 8)
        emb_dim = self.cls.predictions.decoder.weight.size(1)
        self.cls.predictions.decoder.weight = nn.Parameter(
            torch.Tensor(vocab_size, emb_dim))
        self.cls.predictions.bias = nn.Parameter(torch.zeros(vocab_size))
        self.config.vocab_size = vocab_size

    def update_embedding_layer_by_size(self, vocab_size):
        if vocab_size % 8 != 0:
            # pad for tensor cores
            vocab_size += (8 - vocab_size % 8)
        emb_dim = self.cls.predictions.decoder.weight.size(1)
        self.bert.embeddings.word_embeddings = nn.Embedding(
            vocab_size, emb_dim, padding_idx=0)
        self.config.vocab_size = vocab_size

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, output_mask=None, do_padding=True):
        """ only computes masked logits to save some computation"""
        if output_mask is None:
            # reduce to normal forward
            return super().forward(input_ids, token_type_ids, attention_mask,
                                   masked_lm_labels)

        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False)
        # only compute masked outputs
        output_mask = output_mask.byte()
        sequence_output_masked = sequence_output.masked_select(
            output_mask.unsqueeze(-1).expand_as(sequence_output)
        ).contiguous().view(-1, self.config.hidden_size)
        n_pred, hid = sequence_output_masked.size()
        if do_padding and (n_pred == 0 or n_pred % 8):
            # pad for tensor cores
            n_pad = 8 - n_pred % 8
            pad = torch.zeros(n_pad, hid,
                              dtype=sequence_output_masked.dtype,
                              device=sequence_output_masked.device)
            sequence_output_masked = torch.cat(
                [sequence_output_masked, pad], dim=0)
        else:
            n_pad = 0
        prediction_scores = self.cls.predictions(sequence_output_masked)

        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            lm_labels = masked_lm_labels.masked_select(output_mask)
            if n_pad != 0:
                pad = torch.zeros(n_pad,
                                  dtype=lm_labels.dtype,
                                  device=lm_labels.device).fill_(-1)
                lm_labels = torch.cat([lm_labels, pad], dim=0)
            masked_lm_loss = loss_fct(prediction_scores, lm_labels)
            return masked_lm_loss
        else:
            return prediction_scores
