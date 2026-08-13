# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Minimal utils required by EdGE (from DUSt3R misc).

import torch


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            module.requires_grad = False
