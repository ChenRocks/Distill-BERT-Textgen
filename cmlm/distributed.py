"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

distributed utils not presented in OpenNMT-py
"""
import torch.distributed


def broadcast_tensors(tensors, rank=0):
    """ broadcast list of tensors at once
        this can be used to sync parameter initialization across GPUs

    Args:
        tensors: list of Tensors to brodcast
        rank: rank to broadcast
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    sz = sum(t.numel() for t in tensors)
    buffer_t = tensors[0].new(sz).zero_()

    # copy tensors into buffer_t
    offset = 0
    for t in tensors:
        numel = t.numel()
        buffer_t[offset:offset+numel].copy_(t.view(-1))
        offset += numel
    assert offset == sz

    # broadcast
    torch.distributed.broadcast(buffer_t, rank)

    # copy all-reduced buffer back into tensors
    offset = 0
    for t in tensors:
        numel = t.numel()
        t.view(-1).copy_(buffer_t[offset:offset+numel])
        offset += numel
    assert offset == sz == buffer_t.numel()
