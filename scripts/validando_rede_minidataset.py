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

OUTPUT_DIR = Path("resultados_datasetmini")
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

# Não precisamos do weight_loss para inferência
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
print("Número de canais:", model.num_channels)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def read_tif(folder, nome):

    path = folder / f"{nome}.tif"

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def normalize_image(img, p_low=2, p_high=98):

    low = np.percentile(img, p_low)
    high = np.percentile(img, p_high)

    img = (img - low) / (high - low + 1e-8)

    return np.clip(img, 0, 1)


def calcular_metricas(TP, FP, FN, TN):

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

    accuracy = (
        (TP + TN) /
        (TP + FP + FN + TN)
        if (TP + FP + FN + TN) > 0
        else 0
    )

    return precision, recall, f1, iou, accuracy


# ============================================================
# COLORMAP DE DIFERENÇAS
# ============================================================

# 0 = TN
# 1 = TP
# 2 = FP
# 3 = FN

difference_cmap = ListedColormap(
    [
        "black",
        "limegreen",
        "red",
        "dodgerblue"
    ]
)


# ============================================================
# LER DATASET
# ============================================================

df = pd.read_csv(CSV_TEST)

print()
print("============================================")
print("DATASET")
print("============================================")

print("CSV:", CSV_TEST)
print("Número de imagens:", len(df))

print()


# ============================================================
# ACUMULADORES GLOBAIS
# ============================================================

TP_GLOBAL = 0
FP_GLOBAL = 0
FN_GLOBAL = 0
TN_GLOBAL = 0

resultados = []


# ============================================================
# LOOP POR TODAS AS IMAGENS
# ============================================================

for index, row in df.iterrows():

    sample_id = row["id"]

    folder = ROOT / sample_id

    print()
    print("============================================")
    print(f"IMAGEM {index + 1}/{len(df)}")
    print("============================================")

    print("ID:", sample_id)
    print("has_plume:", row["has_plume"])


    # ========================================================
    # LER ENTRADAS
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

    label = read_tif(
        folder,
        "labelbinary"
    )


    # ========================================================
    # MONTAR INPUT
    #
    # [4, H, W]
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


    # numpy -> torch

    x = torch.from_numpy(
        x_np
    ).float()


    # [4,H,W] -> [1,4,H,W]

    x = x.unsqueeze(0)

    x = x.to(device)


    # ========================================================
    # INFERÊNCIA
    # ========================================================

    with torch.no_grad():

        logits = model(x)

        prob = torch.sigmoid(logits)

        pred = (
            prob > 0.5
        ).to(torch.uint8)


    prob_np = (
        prob[0, 0]
        .cpu()
        .numpy()
    )

    pred_np = (
        pred[0, 0]
        .cpu()
        .numpy()
    )

    label_np = (
        label > 0
    ).astype(np.uint8)


    # ========================================================
    # MÉTRICAS
    # ========================================================

    TP = int(
        np.sum(
            (pred_np == 1) &
            (label_np == 1)
        )
    )

    FP = int(
        np.sum(
            (pred_np == 1) &
            (label_np == 0)
        )
    )

    FN = int(
        np.sum(
            (pred_np == 0) &
            (label_np == 1)
        )
    )

    TN = int(
        np.sum(
            (pred_np == 0) &
            (label_np == 0)
        )
    )


    precision, recall, f1, iou, accuracy = (
        calcular_metricas(
            TP,
            FP,
            FN,
            TN
        )
    )


    # ========================================================
    # ACUMULAR GLOBAL
    # ========================================================

    TP_GLOBAL += TP
    FP_GLOBAL += FP
    FN_GLOBAL += FN
    TN_GLOBAL += TN


    # ========================================================
    # SALVAR RESULTADOS DA IMAGEM
    # ========================================================

    resultados.append(
        {
            "id": sample_id,

            "pixels_reais":
                int(label_np.sum()),

            "pixels_previstos":
                int(pred_np.sum()),

            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,

            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,

            "prob_min":
                float(prob_np.min()),

            "prob_max":
                float(prob_np.max())
        }
    )


    # ========================================================
    # PRINT RESULTADO INDIVIDUAL
    # ========================================================

    print(
        "Pixels reais     :",
        int(label_np.sum())
    )

    print(
        "Pixels previstos :",
        int(pred_np.sum())
    )

    print()

    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)

    print()

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"IoU:       {iou:.4f}"
    )


    # ========================================================
    # RGB PARA VISUALIZAÇÃO
    # ========================================================

    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )


    for i in range(3):

        rgb[:, :, i] = (
            normalize_image(
                rgb[:, :, i]
            )
        )


    # ========================================================
    # MAG1C VISUAL
    # ========================================================

    mag1c_vis = normalize_image(
        mag1c,
        p_low=1,
        p_high=99
    )


    # ========================================================
    # MAPA TP / FP / FN
    # ========================================================

    differences = np.zeros_like(
        label_np,
        dtype=np.uint8
    )


    # TP

    differences[
        (pred_np == 1) &
        (label_np == 1)
    ] = 1


    # FP

    differences[
        (pred_np == 1) &
        (label_np == 0)
    ] = 2


    # FN

    differences[
        (pred_np == 0) &
        (label_np == 1)
    ] = 3


    # ========================================================
    # FIGURA
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    axes[0, 0].imshow(rgb)

    axes[0, 0].set_title(
        "AVIRIS RGB",
        fontsize=14
    )

    axes[0, 0].axis("off")


    # --------------------------------------------------------
    # MAG1C
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    axes[0, 2].imshow(
        label_np,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, 2].set_title(
        f"Ground Truth\n"
        f"{label_np.sum()} pixels",
        fontsize=14
    )

    axes[0, 2].axis("off")


    # --------------------------------------------------------
    # PROBABILIDADE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PREDIÇÃO
    # --------------------------------------------------------

    axes[1, 1].imshow(
        pred_np,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1, 1].set_title(
        f"Predição (threshold = 0.5)\n"
        f"{pred_np.sum()} pixels",
        fontsize=14
    )

    axes[1, 1].axis("off")


    # --------------------------------------------------------
    # DIFERENÇAS
    # --------------------------------------------------------

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


    # ========================================================
    # TÍTULO
    # ========================================================

    fig.suptitle(
        f"HyperSTARCOP — {sample_id}\n"
        f"Precision={precision:.4f}   "
        f"Recall={recall:.4f}   "
        f"F1={f1:.4f}   "
        f"IoU={iou:.4f}",
        fontsize=16,
        fontweight="bold"
    )


    plt.tight_layout(
        rect=[0, 0, 1, 0.93]
    )


    # ========================================================
    # SALVAR FIGURA
    # ========================================================

    output_path = (
        OUTPUT_DIR /
        f"validacao_{sample_id}.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )


    # MUITO IMPORTANTE:
    # fecha a figura para não acumular memória
    plt.close(fig)

    print(
        "Figura salva:",
        output_path
    )


# ============================================================
# RESULTADOS POR IMAGEM
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)


csv_individual = (
    OUTPUT_DIR /
    "metricas_por_imagem.csv"
)

resultados_df.to_csv(
    csv_individual,
    index=False
)


# ============================================================
# MÉTRICAS GLOBAIS
# ============================================================

precision_global, recall_global, f1_global, iou_global, accuracy_global = (
    calcular_metricas(
        TP_GLOBAL,
        FP_GLOBAL,
        FN_GLOBAL,
        TN_GLOBAL
    )
)


# ============================================================
# MÉDIA DAS MÉTRICAS INDIVIDUAIS
#
# Isso é diferente da métrica global.
# ============================================================

precision_media = (
    resultados_df["precision"].mean()
)

recall_media = (
    resultados_df["recall"].mean()
)

f1_media = (
    resultados_df["f1"].mean()
)

iou_media = (
    resultados_df["iou"].mean()
)


# ============================================================
# RESUMO GLOBAL
# ============================================================

print()
print()
print("====================================================")
print("RESULTADO GLOBAL - TODOS OS PIXELS")
print("====================================================")

print()

print("Número de imagens:", len(df))

print()

print("TP:", TP_GLOBAL)
print("FP:", FP_GLOBAL)
print("FN:", FN_GLOBAL)
print("TN:", TN_GLOBAL)

print()

print(
    f"Precision: {precision_global:.4f}"
)

print(
    f"Recall:    {recall_global:.4f}"
)

print(
    f"F1:        {f1_global:.4f}"
)

print(
    f"IoU:       {iou_global:.4f}"
)

print(
    f"Accuracy:  {accuracy_global:.4f}"
)


print()
print("====================================================")
print("MÉDIA DAS IMAGENS")
print("====================================================")

print(
    f"Precision média: {precision_media:.4f}"
)

print(
    f"Recall médio:    {recall_media:.4f}"
)

print(
    f"F1 médio:        {f1_media:.4f}"
)

print(
    f"IoU médio:       {iou_media:.4f}"
)


# ============================================================
# SALVAR RESUMO
# ============================================================

resumo_global = pd.DataFrame(
    [
        {
            "num_imagens":
                len(df),

            "TP":
                TP_GLOBAL,

            "FP":
                FP_GLOBAL,

            "FN":
                FN_GLOBAL,

            "TN":
                TN_GLOBAL,

            "precision_global":
                precision_global,

            "recall_global":
                recall_global,

            "f1_global":
                f1_global,

            "iou_global":
                iou_global,

            "accuracy_global":
                accuracy_global,

            "precision_media":
                precision_media,

            "recall_media":
                recall_media,

            "f1_media":
                f1_media,

            "iou_media":
                iou_media,
        }
    ]
)


csv_global = (
    OUTPUT_DIR /
    "metricas_globais.csv"
)

resumo_global.to_csv(
    csv_global,
    index=False
)


print()
print("====================================================")
print("ARQUIVOS GERADOS")
print("====================================================")

print(
    "Métricas por imagem:",
    csv_individual
)

print(
    "Métricas globais:",
    csv_global
)

print(
    "Figuras:",
    OUTPUT_DIR
)