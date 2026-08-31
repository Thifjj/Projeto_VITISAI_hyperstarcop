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
# ISOLAR SOMENTE A CNN
# ============================================================

network = model.network

network = network.to(device)
network.eval()


print()
print("============================================")
print("MODELO ISOLADO")
print("============================================")

print("ModelModule:")
print(type(model))

print()

print("CNN:")
print(type(network))


# ============================================================
# FUNÇÃO PARA LER TIFF
# ============================================================

def read_tif(folder, nome):

    path = folder / f"{nome}.tif"

    with rasterio.open(path) as src:

        return src.read(1).astype(
            np.float32
        )


# ============================================================
# DATASET
# ============================================================

df = pd.read_csv(CSV_TEST)


print()
print("============================================")
print("DATASET")
print("============================================")

print("Número de imagens:", len(df))


# ============================================================
# ACUMULADORES
# ============================================================

maior_diferenca_global = 0.0

soma_diferenca = 0.0

total_elementos = 0

predicoes_diferentes_total = 0

resultados = []


# ============================================================
# LOOP
# ============================================================

for index, row in df.iterrows():

    sample_id = row["id"]

    folder = ROOT / sample_id


    print()
    print("============================================")
    print(
        f"IMAGEM {index + 1}/{len(df)}"
    )
    print("============================================")

    print("ID:", sample_id)


    # ========================================================
    # CARREGAR OS 4 CANAIS
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
    # MONTAR INPUT
    #
    # [4,512,512]
    # ========================================================

    x_np = np.stack(
        [
            mag1c,
            red,
            green,
            blue
        ],
        axis=0
    )


    x = torch.from_numpy(
        x_np
    ).float()


    # ========================================================
    # [4,H,W] -> [1,4,H,W]
    # ========================================================

    x = x.unsqueeze(0)

    x = x.to(device)


    # ========================================================
    # CAMINHO A
    #
    # MODELO ORIGINAL
    #
    # model(x)
    #
    # Internamente:
    #
    # normalizer -> CNN
    # ========================================================

    with torch.no_grad():

        logits_original = model(x)


    # ========================================================
    # CAMINHO B
    #
    # NORMALIZAÇÃO SEPARADA
    # ========================================================

    with torch.no_grad():

        x_normalized = (
            model.normalizer.normalize_x(x)
        )


    # ========================================================
    # SOMENTE A CNN
    # ========================================================

    with torch.no_grad():

        logits_network = network(
            x_normalized
        )


    # ========================================================
    # COMPARAÇÃO DOS LOGITS
    # ========================================================

    diff = torch.abs(
        logits_original -
        logits_network
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

    prob_network = torch.sigmoid(
        logits_network
    )


    prob_diff = torch.abs(
        prob_original -
        prob_network
    )


    max_prob_diff = (
        prob_diff.max().item()
    )


    # ========================================================
    # MÁSCARAS BINÁRIAS
    # ========================================================

    pred_original = (
        prob_original > 0.5
    ).to(torch.uint8)

    pred_network = (
        prob_network > 0.5
    ).to(torch.uint8)


    pixels_diferentes = (
        pred_original != pred_network
    ).sum().item()


    # ========================================================
    # ESTATÍSTICAS GLOBAIS
    # ========================================================

    maior_diferenca_global = max(
        maior_diferenca_global,
        max_diff
    )

    soma_diferenca += (
        diff.sum().item()
    )

    total_elementos += (
        diff.numel()
    )

    predicoes_diferentes_total += (
        pixels_diferentes
    )


    # ========================================================
    # RESULTADO DESTA IMAGEM
    # ========================================================

    resultados.append(
        {
            "id":
                sample_id,

            "max_diff_logits":
                max_diff,

            "mean_diff_logits":
                mean_diff,

            "max_diff_prob":
                max_prob_diff,

            "pixels_pred_diferentes":
                pixels_diferentes,
        }
    )


    print(
        "Input original:",
        tuple(x.shape)
    )

    print(
        "Input normalizado:",
        tuple(x_normalized.shape)
    )

    print(
        "Logits original:",
        tuple(logits_original.shape)
    )

    print(
        "Logits CNN pura:",
        tuple(logits_network.shape)
    )

    print()

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
        max_prob_diff
    )

    print(
        "Pixels de predição diferentes:",
        pixels_diferentes
    )


# ============================================================
# RESULTADO GLOBAL
# ============================================================

diferenca_media_global = (
    soma_diferenca /
    total_elementos
)


print()
print()
print("================================================")
print("RESULTADO FINAL")
print("================================================")

print(
    "Imagens testadas:",
    len(df)
)

print()

print(
    "Maior diferença encontrada nos logits:",
    maior_diferenca_global
)

print(
    "Diferença média global:",
    diferenca_media_global
)

print()

print(
    "Pixels de classificação diferentes:",
    predicoes_diferentes_total
)


# ============================================================
# VERIFICAÇÃO
# ============================================================

if (
    maior_diferenca_global == 0.0
    and
    predicoes_diferentes_total == 0
):

    print()
    print("============================================")
    print("RESULTADO: MODELOS IDÊNTICOS")
    print("============================================")

    print()
    print(
        "model(x) == "
        "model.network(model.normalizer.normalize_x(x))"
    )

else:

    print()
    print("============================================")
    print("ATENÇÃO: EXISTEM DIFERENÇAS")
    print("============================================")


# ============================================================
# SALVAR CSV
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)

output_csv = Path(
    "vitis_ai/results/comparacao_modelo_original_vs_rede.csv"
)

output_csv.parent.mkdir(
    parents=True,
    exist_ok=True
)

resultados_df.to_csv(
    output_csv,
    index=False
)


print()
print(
    "Resultados salvos em:",
    output_csv
)