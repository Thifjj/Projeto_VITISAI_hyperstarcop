import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

import starcop
from starcop.models.model_module import ModelModule


# ============================================================
# CAMINHOS
# ============================================================

CHECKPOINT = "model/final_checkpoint_model.ckpt"

OUTPUT_DIR = Path("vitis_ai/float_model")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_WEIGHTS = (
    OUTPUT_DIR /
    "hyperstarcop_network_fp32.pth"
)


# ============================================================
# CONFIG
# ============================================================

STARCOP_DIR = os.path.dirname(
    starcop.__file__
)

CONFIG_BASE = os.path.join(
    STARCOP_DIR,
    "config.yaml"
)

config = OmegaConf.load(
    CONFIG_BASE
)


config.dataset.input_products = [
    "mag1c",
    "TOA_AVIRIS_640nm",
    "TOA_AVIRIS_550nm",
    "TOA_AVIRIS_460nm",
]

config.dataset.output_products = [
    "labelbinary"
]

config.dataset.use_weight_loss = False


config.model.train = False
config.model.test = False

config.model.model_mode = (
    "segmentation_output"
)

config.model.model_type = (
    "unet_semseg"
)

config.model.semseg_backbone = (
    "mobilenet_v2"
)

config.model.num_classes = 1

config.model.optimizer = "adam"

config.model.lr = 0.0001
config.model.lr_decay = 0.5
config.model.lr_patience = 4

config.model.loss = (
    "BCEWithLogitsLoss"
)

config.model.pos_weight = 1

config.model.early_stopping_patience = 8


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# CARREGAR CHECKPOINT ORIGINAL
# ============================================================

model = ModelModule.load_from_checkpoint(
    CHECKPOINT,
    settings=config,
    map_location=device,
    weights_only=False,
)

model.eval()


# ============================================================
# PEGAR SOMENTE A CNN
# ============================================================

network = model.network

network = network.cpu()
network.eval()


print()
print("============================================")
print("REDE EXTRAÍDA")
print("============================================")

print(type(network))


# ============================================================
# CONTAR PARÂMETROS
# ============================================================

total_params = sum(
    p.numel()
    for p in network.parameters()
)

trainable_params = sum(
    p.numel()
    for p in network.parameters()
    if p.requires_grad
)


print()
print("Parâmetros totais:", total_params)
print(
    "Parâmetros treináveis:",
    trainable_params
)


# ============================================================
# SALVAR SOMENTE STATE_DICT
# ============================================================

torch.save(
    network.state_dict(),
    OUTPUT_WEIGHTS
)


print()
print("============================================")
print("MODELO PURO SALVO")
print("============================================")

print(
    "Arquivo:",
    OUTPUT_WEIGHTS
)

print(
    "Tamanho:",
    OUTPUT_WEIGHTS.stat().st_size,
    "bytes"
)