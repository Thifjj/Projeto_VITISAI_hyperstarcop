
import torch
import executorch
from importlib.metadata import version
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

print('PyTorch:', torch.__version__)
print('ExecuTorch:', version('executorch'))
print('XNNPACK: disponível')