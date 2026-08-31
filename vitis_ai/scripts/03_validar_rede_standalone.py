import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import segmentation_models_pytorch as smp

from omegaconf import OmegaConf

import starcop
from starcop.models.model_module import ModelModule


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path("STARCOP_mini")
CSV_TEST = ROOT / "test_mini10.csv"

CHECKPOINT_ORIGINAL = Path(
    "model/final_checkpoint_model.ckpt"
)

WEIGHTS_STANDALONE = Path(
    "vitis_ai/float_model/hyperstarcop_network_fp32.pth"
)

OUTPUT_DIR = Path(
    "vitis_ai/results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# CONFIGURAÇÃO DO MODELO ORIGINAL
#
# Só precisamos disso para carregar o checkpoint original
# e usá-lo como referência.
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
# 1. CARREGAR MODELO ORIGINAL
# ============================================================

print()
print("============================================")
print("CARREGANDO MODELO ORIGINAL")
print("============================================")

model_original = ModelModule.load_from_checkpoint(
    CHECKPOINT_ORIGINAL,
    settings=config,
    map_location=device,
    weights_only=False,
)

model_original = model_original.to(
    device
)

model_original.eval()

print("Modelo original carregado.")


# ============================================================
# 2. RECONSTRUIR REDE STANDALONE
#
# IMPORTANTE:
#
# Aqui não usamos:
#
# ModelModule
# Lightning
# DataNormalizer
#
# É somente a CNN.
# ============================================================

print()
print("============================================")
print("CRIANDO REDE STANDALONE")
print("============================================")

model_standalone = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None,
)


# ============================================================
# 3. CARREGAR .PTH
# ============================================================

state_dict = torch.load(
    WEIGHTS_STANDALONE,
    map_location="cpu",
    weights_only=True,
)

resultado_load = (
    model_standalone.load_state_dict(
        state_dict,
        strict=True,
    )
)


print("Pesos carregados:")
print(resultado_load)


model_standalone = (
    model_standalone
    .to(device)
)

model_standalone.eval()


# ============================================================
# PARÂMETROS
# ============================================================

params = sum(
    p.numel()
    for p in model_standalone.parameters()
)


print()
print("Parâmetros standalone:", params)


# ============================================================
# LEITURA TIFF
# ============================================================

def read_tif(folder, nome):

    path = (
        folder /
        f"{nome}.tif"
    )

    with rasterio.open(path) as src:

        return src.read(1).astype(
            np.float32
        )


# ============================================================
# NORMALIZAÇÃO MANUAL
#
# Exatamente equivalente ao DataNormalizer original:
#
# mag1c  / 1750
# RGB    / 60
# clip   [0, 2]
# ============================================================

def normalizar_manual(x):

    # clone para não alterar a entrada original
    x = x.clone()

    x[:, 0] = torch.clamp(
        x[:, 0] / 1750.0,
        min=0.0,
        max=2.0
    )

    x[:, 1] = torch.clamp(
        x[:, 1] / 60.0,
        min=0.0,
        max=2.0
    )

    x[:, 2] = torch.clamp(
        x[:, 2] / 60.0,
        min=0.0,
        max=2.0
    )

    x[:, 3] = torch.clamp(
        x[:, 3] / 60.0,
        min=0.0,
        max=2.0
    )

    return x


# ============================================================
# DATASET
# ============================================================

df = pd.read_csv(
    CSV_TEST
)


print()
print("============================================")
print("DATASET")
print("============================================")

print(
    "Imagens:",
    len(df)
)


# ============================================================
# ACUMULADORES
# ============================================================

resultados = []

max_diff_global = 0.0
soma_diff_global = 0.0
total_elementos = 0

pixels_diferentes_global = 0

max_diff_normalizacao_global = 0.0


# ============================================================
# LOOP POR TODAS AS IMAGENS
# ============================================================

for index, row in df.iterrows():

    sample_id = row["id"]

    folder = (
        ROOT /
        sample_id
    )


    print()
    print(
        "============================================"
    )

    print(
        f"IMAGEM {index + 1}/{len(df)}"
    )

    print(
        "============================================"
    )

    print(
        "ID:",
        sample_id
    )


    # ========================================================
    # CARREGAR CANAIS
    # ========================================================

    mag1c = read_tif(
        folder,
        "mag1c"
    )

    red = read_tif(
        folder,
        "TOA_AVIRIS_640nm"
    )

    green = read_tif(
        folder,
        "TOA_AVIRIS_550nm"
    )

    blue = read_tif(
        folder,
        "TOA_AVIRIS_460nm"
    )


    # ========================================================
    # STACK
    #
    # [4,512,512]
    # ========================================================

    x_np = np.stack(
        [
            mag1c,
            red,
            green,
            blue,
        ],
        axis=0,
    )


    x = torch.from_numpy(
        x_np
    ).float()


    # ========================================================
    # BATCH
    #
    # [1,4,512,512]
    # ========================================================

    x = (
        x
        .unsqueeze(0)
        .to(device)
    )


    # ========================================================
    # CAMINHO ORIGINAL
    #
    # ModelModule:
    #
    # normalizer
    #    ↓
    # network
    # ========================================================

    with torch.no_grad():

        logits_original = (
            model_original(x)
        )


    # ========================================================
    # NORMALIZAÇÃO ORIGINAL
    #
    # Apenas para comparar com nossa normalização manual.
    # ========================================================

    with torch.no_grad():

        x_norm_original = (
            model_original
            .normalizer
            .normalize_x(x)
        )


    # ========================================================
    # NORMALIZAÇÃO MANUAL
    # ========================================================

    x_norm_manual = (
        normalizar_manual(x)
    )


    # ========================================================
    # COMPARAR NORMALIZAÇÃO
    # ========================================================

    diff_norm = torch.abs(
        x_norm_original -
        x_norm_manual
    )

    max_diff_norm = (
        diff_norm
        .max()
        .item()
    )


    max_diff_normalizacao_global = max(
        max_diff_normalizacao_global,
        max_diff_norm
    )


    # ========================================================
    # REDE STANDALONE
    #
    # Aqui NÃO usamos ModelModule.
    # ========================================================

    with torch.no_grad():

        logits_standalone = (
            model_standalone(
                x_norm_manual
            )
        )


    # ========================================================
    # COMPARAÇÃO DOS LOGITS
    # ========================================================

    diff = torch.abs(
        logits_original -
        logits_standalone
    )


    max_diff = (
        diff.max().item()
    )

    mean_diff = (
        diff.mean().item()
    )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    prob_original = torch.sigmoid(
        logits_original
    )

    prob_standalone = torch.sigmoid(
        logits_standalone
    )


    diff_prob = torch.abs(
        prob_original -
        prob_standalone
    )


    max_diff_prob = (
        diff_prob
        .max()
        .item()
    )


    # ========================================================
    # PREDIÇÕES BINÁRIAS
    # ========================================================

    pred_original = (
        prob_original > 0.5
    )

    pred_standalone = (
        prob_standalone > 0.5
    )


    pixels_diferentes = (
        pred_original !=
        pred_standalone
    ).sum().item()


    # ========================================================
    # ACUMULADORES
    # ========================================================

    max_diff_global = max(
        max_diff_global,
        max_diff
    )


    soma_diff_global += (
        diff.sum().item()
    )


    total_elementos += (
        diff.numel()
    )


    pixels_diferentes_global += (
        pixels_diferentes
    )


    # ========================================================
    # SALVAR RESULTADO
    # ========================================================

    resultados.append(
        {
            "id":
                sample_id,

            "max_diff_normalizacao":
                max_diff_norm,

            "max_diff_logits":
                max_diff,

            "mean_diff_logits":
                mean_diff,

            "max_diff_prob":
                max_diff_prob,

            "pixels_pred_diferentes":
                pixels_diferentes,
        }
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        "Diferença normalização:",
        max_diff_norm
    )

    print(
        "Diferença máxima logits:",
        max_diff
    )

    print(
        "Diferença média logits:",
        mean_diff
    )

    print(
        "Diferença máxima prob:",
        max_diff_prob
    )

    print(
        "Pixels diferentes:",
        pixels_diferentes
    )


# ============================================================
# RESULTADO GLOBAL
# ============================================================

mean_diff_global = (
    soma_diff_global /
    total_elementos
)


print()
print()
print(
    "================================================"
)

print(
    "RESULTADO FINAL"
)

print(
    "================================================"
)


print()

print(
    "Imagens testadas:",
    len(df)
)

print()

print(
    "Maior diferença na normalização:",
    max_diff_normalizacao_global
)

print()

print(
    "Maior diferença nos logits:",
    max_diff_global
)

print(
    "Diferença média global:",
    mean_diff_global
)

print()

print(
    "Pixels de predição diferentes:",
    pixels_diferentes_global
)


# ============================================================
# VERIFICAÇÃO FINAL
# ============================================================

if (
    max_diff_normalizacao_global == 0.0
    and
    max_diff_global == 0.0
    and
    pixels_diferentes_global == 0
):

    print()
    print(
        "============================================"
    )

    print(
        "RESULTADO: STANDALONE 100% IDÊNTICO"
    )

    print(
        "============================================"
    )

    print()

    print(
        "Checkpoint original"
    )

    print(
        "        ="
    )

    print(
        "U-Net standalone + .pth"
    )


else:

    print()
    print(
        "============================================"
    )

    print(
        "ATENÇÃO: EXISTEM DIFERENÇAS"
    )

    print(
        "============================================"
    )


# ============================================================
# CSV
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


output_csv = (
    OUTPUT_DIR /
    "comparacao_standalone.csv"
)


resultados_df.to_csv(
    output_csv,
    index=False
)


print()
print(
    "CSV salvo em:",
    output_csv
)