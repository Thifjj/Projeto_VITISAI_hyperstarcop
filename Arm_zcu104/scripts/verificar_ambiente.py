#!/usr/bin/env python3
"""Confere as dependências de exportação ExecuTorch/XNNPACK."""

from importlib.metadata import version

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


print("PyTorch:", torch.__version__)
print("ExecuTorch:", version("executorch"))
print("XNNPACK:", XnnpackPartitioner.__name__, "disponível")
