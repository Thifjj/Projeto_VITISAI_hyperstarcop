import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf

import starcop
from starcop.models.model_module import ModelModule


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path("STARCOP_mini")
CSV_TEST = ROOT / "test_mini10.csv"

CHECKPOINT = "model/final_checkpoint_model.ckpt"

OUTPUT_DIR = Path("resultados")
OUTPUT_DIR.mkdir(exist_ok=True)

STARCOP_DIR = os.path.dirname(starcop.__file__)
CONFIG_BASE = os.path.join(STARCOP_DIR, "config.yaml")


# ============================================================
# CONFIGURAÇÃO DO HYPERSTARCOP
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
# CARREGAR MODELO
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


# ============================================================
# ESCOLHER AMOSTRA
# ============================================================

df = pd.read_csv(CSV_TEST)

row = df.iloc[0]

sample_id = row["id"]
folder = ROOT / sample_id

print("Amostra:", sample_id)


# ============================================================
# LEITURA TIFF
# ============================================================

def read_tif(nome):

    path = folder / f"{nome}.tif"

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


mag1c = read_tif("mag1c")

red = read_tif("TOA_AVIRIS_640nm")
green = read_tif("TOA_AVIRIS_550nm")
blue = read_tif("TOA_AVIRIS_460nm")

label = read_tif("labelbinary")


# ============================================================
# MONTAR ENTRADA
# ============================================================

x_np = np.stack(
    [
        mag1c,
        red,
        green,
        blue
    ],
    axis=0
)

x = torch.from_numpy(x_np).float()

x = x.unsqueeze(0).to(device)


# ============================================================
# INFERÊNCIA
# ============================================================

with torch.no_grad():

    logits = model(x)

    prob = torch.sigmoid(logits)

    pred = (prob > 0.5).to(torch.uint8)


prob_np = prob[0, 0].cpu().numpy()

pred_np = pred[0, 0].cpu().numpy()

label_np = (label > 0).astype(np.uint8)


# ============================================================
# MÉTRICAS
# ============================================================

TP = np.sum(
    (pred_np == 1) &
    (label_np == 1)
)

FP = np.sum(
    (pred_np == 1) &
    (label_np == 0)
)

FN = np.sum(
    (pred_np == 0) &
    (label_np == 1)
)

TN = np.sum(
    (pred_np == 0) &
    (label_np == 0)
)


precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)

iou = (
    TP / (TP + FP + FN)
    if (TP + FP + FN) > 0
    else 0
)


# ============================================================
# FUNÇÃO DE NORMALIZAÇÃO PARA VISUALIZAÇÃO
# ============================================================

def normalize_image(img, p_low=2, p_high=98):

    low = np.percentile(img, p_low)
    high = np.percentile(img, p_high)

    img = (img - low) / (high - low + 1e-8)

    return np.clip(img, 0, 1)


# ============================================================
# RGB AVIRIS
# ============================================================

rgb = np.stack(
    [
        red,
        green,
        blue
    ],
    axis=-1
)

# Normalizar cada canal separadamente

for i in range(3):
    rgb[:, :, i] = normalize_image(
        rgb[:, :, i]
    )


# ============================================================
# MAG1C PARA VISUALIZAÇÃO
# ============================================================

mag1c_vis = normalize_image(
    mag1c,
    p_low=1,
    p_high=99
)


# ============================================================
# MAPA DE DIFERENÇAS
#
# 0 = TN
# 1 = TP
# 2 = FP
# 3 = FN
# ============================================================

differences = np.zeros_like(
    label_np,
    dtype=np.uint8
)

differences[
    (pred_np == 1) &
    (label_np == 1)
] = 1

differences[
    (pred_np == 1) &
    (label_np == 0)
] = 2

differences[
    (pred_np == 0) &
    (label_np == 1)
] = 3


# TN = preto
# TP = verde
# FP = vermelho
# FN = azul

difference_cmap = ListedColormap(
    [
        "black",
        "limegreen",
        "red",
        "dodgerblue"
    ]
)


# ============================================================
# FIGURA
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)


# ------------------------------------------------------------
# RGB
# ------------------------------------------------------------

axes[0, 0].imshow(rgb)

axes[0, 0].set_title(
    "AVIRIS RGB",
    fontsize=14
)

axes[0, 0].axis("off")


# ------------------------------------------------------------
# MAG1C
# ------------------------------------------------------------

im_mag = axes[0, 1].imshow(
    mag1c_vis,
    cmap="magma"
)

axes[0, 1].set_title(
    "MAG1C",
    fontsize=14
)

axes[0, 1].axis("off")

fig.colorbar(
    im_mag,
    ax=axes[0, 1],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

axes[0, 2].imshow(
    label_np,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 2].set_title(
    f"Ground Truth\n"
    f"{label_np.sum()} pixels de metano",
    fontsize=14
)

axes[0, 2].axis("off")


# ------------------------------------------------------------
# PROBABILIDADE
# ------------------------------------------------------------

im_prob = axes[1, 0].imshow(
    prob_np,
    cmap="inferno",
    vmin=0,
    vmax=1
)

axes[1, 0].set_title(
    "Probabilidade de metano",
    fontsize=14
)

axes[1, 0].axis("off")

fig.colorbar(
    im_prob,
    ax=axes[1, 0],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------------
# PREDIÇÃO
# ------------------------------------------------------------

axes[1, 1].imshow(
    pred_np,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1, 1].set_title(
    f"Predição (threshold = 0.5)\n"
    f"{pred_np.sum()} pixels previstos",
    fontsize=14
)

axes[1, 1].axis("off")


# ------------------------------------------------------------
# DIFERENÇAS
# ------------------------------------------------------------

axes[1, 2].imshow(
    differences,
    cmap=difference_cmap,
    vmin=0,
    vmax=3
)

axes[1, 2].set_title(
    "Comparação\n"
    "Verde=TP | Vermelho=FP | Azul=FN",
    fontsize=14
)

axes[1, 2].axis("off")


# ============================================================
# TÍTULO GERAL
# ============================================================

fig.suptitle(
    f"HyperSTARCOP — {sample_id}\n"
    f"Precision={precision:.4f}   "
    f"Recall={recall:.4f}   "
    f"F1={f1:.4f}   "
    f"IoU={iou:.4f}",
    fontsize=17,
    fontweight="bold"
)


plt.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ============================================================
# SALVAR
# ============================================================

output_path = (
    OUTPUT_DIR /
    f"validacao_{sample_id}.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


print()
print("===================================")
print("RESULTADOS")
print("===================================")

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
print("Figura salva em:")
print(output_path)