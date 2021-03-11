# MIT License
#
# Copyright (c) 2018 Jonathan Lorraine

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torch.distributions.utils import probs_to_logits


def split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


def gumbel_binary(theta, temperature=0.5, hard=False):
    """theta is a vector of unnormalized probabilities
    Returns:
        A vector that becomes binary as the temperature --> 0
    """
    u = Variable(torch.rand(theta.size()))
    z = theta + torch.log(u / (1 - u))
    a = F.sigmoid(z / temperature)

    if hard:
        a_hard = torch.round(a)
        return (a_hard - a).detach() + a
    else:
        return a


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature, dim=-1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=dim)


def gumbel_softmax(probs, temperature, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    logits = probs_to_logits(probs)
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    if hard:
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
    else:
        return y


def sample_conditional_concrete(probs, choice, temperature):
    """
    Arguments:
        probs: (K,)   --> probs: (batch_size, K)
        choice: (batch_size,)

    Returns:
        cond_soft_sample: (batch_size, K)
    """
    # uniforms = Variable(probs.data.new(choice.size(0), probs.size(0)).uniform_())
    uniforms = Variable(probs.data.new(probs.size()).uniform_())
    cond_concrete_sample = -torch.log(-torch.log(uniforms)/probs - torch.log(torch.gather(uniforms, dim=1, index=choice.unsqueeze(1))))
    gumbels = -torch.log(-torch.log(torch.gather(uniforms, dim=1, index=choice.unsqueeze(1))))
    cond_concrete_sample = cond_concrete_sample.scatter(1, choice.unsqueeze(1), gumbels)
    cond_soft_sample = F.softmax(cond_concrete_sample / temperature, dim=1)
    return cond_soft_sample


def st(x):
    shape = x.size()
    _, ind = x.max(dim=-1)
    x_hard = torch.zeros_like(x).view(-1, shape[-1])
    x_hard.scatter_(1, ind.view(-1, 1), 1)
    x_hard = x_hard.view(*shape)
    return (x_hard - x).detach() + x


def sample_one_hot(dist):
    """Samples a one-hot vector from a multinomial distribution parameterized by dist.
    """
    shape = dist.size()
    sample = torch.multinomial(dist)
    one_hot = torch.zeros_like(dist).view(-1, shape[-1])
    one_hot.scatter_(1, sample.view(-1, 1), 1)
    one_hot = one_hot.view(*shape)
    return one_hot


def tanh_hard(theta, hard=True):
    """theta is a vector of unnormalized probabilities
    Returns:
        A vector that becomes binary as the temperature --> 0
    """
    a = F.tanh(theta)

    if hard:
        a_hard = a.clone().detach()
        # a_hard = Variable(a.data)
        # a_hard.apply_(lambda x: -1 if x < 0 else 1)
        a_hard.data.apply_(lambda x: -1 if x < 0 else 1)
        return (a_hard - a).detach() + a
    else:
        return a


def sigmoid_hard(theta, hard=True):
    """theta is a vector of unnormalized probabilities
    Returns:
        A binary vector (each element in {0, 1}), where the gradients
        are computed using the sigmoid continuous relaxation (in (0,1)).
    """
    a = F.sigmoid(theta)

    if hard:
        a_hard = torch.round(a)
        return (a_hard - a).detach() + a
    else:
        return a


def st_softmax(logits, temperature=1, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = F.softmax(logits / temperature, dim=-1)
    shape = y.size()
    if hard:
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
    else:
        return y
