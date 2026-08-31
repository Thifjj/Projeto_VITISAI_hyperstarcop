import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch

from omegaconf import OmegaConf

import starcop
from starcop.models.model_module import ModelModule


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path("STARCOP_mini")
CSV_TEST = ROOT / "test_mini10.csv"

CHECKPOINT = "model/final_checkpoint_model.ckpt"

STARCOP_DIR = os.path.dirname(starcop.__file__)
CONFIG_BASE = os.path.join(STARCOP_DIR, "config.yaml")


# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

config = OmegaConf.load(CONFIG_BASE)

config.dataset.input_products = [
    "mag1c",
    "TOA_AVIRIS_640nm",
    "TOA_AVIRIS_550nm",
    "TOA_AVIRIS_460nm",
]

config.dataset.output_products = ["labelbinary"]
config.dataset.use_weight_loss = False

config.model.train = False
config.model.test = False

config.model.model_mode = "segmentation_output"
config.model.model_type = "unet_semseg"
config.model.semseg_backbone = "mobilenet_v2"
config.model.num_classes = 1

config.model.optimizer = "adam"
config.model.lr = 0.0001
config.model.lr_decay = 0.5
config.model.lr_patience = 4

config.model.loss = "BCEWithLogitsLoss"
config.model.pos_weight = 1
config.model.early_stopping_patience = 8


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CARREGAR CHECKPOINT
# ============================================================

model = ModelModule.load_from_checkpoint(
    CHECKPOINT,
    settings=config,
    map_location=device,
    weights_only=False,
)

model = model.to(device)
model.eval()

print("Checkpoint carregado.")
print("Número de canais:", model.num_channels)


# ============================================================
# ESCOLHER PRIMEIRA AMOSTRA
# ============================================================

df = pd.read_csv(CSV_TEST)

row = df.iloc[0]

sample_id = row["id"]
folder = ROOT / sample_id

print("\n===================================")
print("AMOSTRA")
print("===================================")

print("ID:", sample_id)
print("has_plume:", row["has_plume"])
print("Pasta:", folder)


# ============================================================
# LEITURA DOS TIFF
# ============================================================

def read_tif(nome):
    path = folder / f"{nome}.tif"

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


mag1c = read_tif("mag1c")
r = read_tif("TOA_AVIRIS_640nm")
g = read_tif("TOA_AVIRIS_550nm")
b = read_tif("TOA_AVIRIS_460nm")

label = read_tif("labelbinary")


print("\nShapes:")
print("mag1c :", mag1c.shape)
print("640nm :", r.shape)
print("550nm :", g.shape)
print("460nm :", b.shape)
print("label :", label.shape)


# ============================================================
# MONTAR TENSOR DE ENTRADA
# ============================================================

x = np.stack(
    [
        mag1c,
        r,
        g,
        b,
    ],
    axis=0,
)

x = torch.from_numpy(x).float()

# [4,512,512] -> [1,4,512,512]
x = x.unsqueeze(0).to(device)


print("\nEntrada:", x.shape)


# ============================================================
# INFERÊNCIA
# ============================================================

with torch.no_grad():

    logits = model(x)

    prob = torch.sigmoid(logits)

    pred = (prob > 0.5).to(torch.uint8)


print("Logits :", logits.shape)
print("Prob   :", prob.shape)
print("Pred   :", pred.shape)


# ============================================================
# GROUND TRUTH
# ============================================================

label = torch.from_numpy(
    (label > 0).astype(np.uint8)
).to(device)

pred2d = pred[0, 0]


# ============================================================
# MÉTRICAS
# ============================================================

TP = ((pred2d == 1) & (label == 1)).sum().item()
FP = ((pred2d == 1) & (label == 0)).sum().item()
FN = ((pred2d == 0) & (label == 1)).sum().item()
TN = ((pred2d == 0) & (label == 0)).sum().item()


precision = TP / (TP + FP) if (TP + FP) else 0

recall = TP / (TP + FN) if (TP + FN) else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall)
    else 0
)

iou = (
    TP / (TP + FP + FN)
    if (TP + FP + FN)
    else 0
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n===================================")
print("RESULTADOS")
print("===================================")

print("Pixels reais:", label.sum().item())
print("Pixels previstos:", pred2d.sum().item())

print()

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)

print()

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"IoU:       {iou:.4f}")

print()

print("Prob min:", prob.min().item())
print("Prob max:", prob.max().item())