"""
Utilities
"""
import torch
import os
import argparse


def create_folder_when_necessary(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def accuracy_score(y_pred, y):
    """calculate accuracy"""
    # type: # (Tensor, Tensor) -> float
    return int(torch.sum((y == y_pred))) / len(y)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def sigmoid_derivative(y):
    # type: (Tensor) -> Tensor
    return y * (1 - y)


def tanh_derivative(y):
    # type: (Tensor) -> Tensor
    return 1 - y ** 2


def save_grad_dict(grads, name):
    # save grads of non-leaf node as dictionsary
    def hook(grad):
        grads[name] = grad

    return hook


def save_grad_tensor(grads, i):
    # save grads of non-leaf node as tensor
    def hook(grad):
        grads[i] = grad

    return hook


def save_grad_var(var):
    def hook(grad):
        var.grad = grad

    return hook


def extract_optim_name(opt):
    optstr = str(opt)
    optstr = optstr[optstr.find("optim.") + 6:-2]
    return optstr[len(optstr) // 2 + 1:]


def one_hot_encoded(labels_seqs, batch_size, seq, output_size):
    idx = (torch.arange(output_size)).repeat(seq * batch_size, 1)  # [seq * B x O]
    labels_repeated = torch.repeat_interleave(labels_seqs[:, None], output_size, dim=1)  # [seq * B x O]
    return (labels_repeated == idx).float()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
