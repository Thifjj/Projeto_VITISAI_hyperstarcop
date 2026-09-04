#!/usr/bin/env python3
"""Compara o modelo PyTorch FP32 com o PTE XNNPACK no STARCOP_mini."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import segmentation_models_pytorch as smp

from executorch.runtime import Runtime


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path("STARCOP_mini")

CSV_TEST = ROOT / "test_mini10.csv"

FLOAT_MODEL = Path(
    "vitis_ai/float_model/hyperstarcop_network_fp32.pth"
)

PTE_MODEL = Path(
    "Arm_zcu104/model/hyperstarcop_xnnpack_fp32.pte"
)

OUTPUT_DIR = Path(
    "Arm_zcu104/reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# O PTE / ExecuTorch será executado na CPU.
device = torch.device("cpu")

print("Device:", device)


# ============================================================
# RECONSTRUIR MODELO FP32
# ============================================================

print()
print("============================================")
print("CARREGANDO MODELO FP32")
print("============================================")

model_float = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None,
)

state_dict = torch.load(
    FLOAT_MODEL,
    map_location="cpu",
    weights_only=True,
)

resultado_load = model_float.load_state_dict(
    state_dict,
    strict=True,
)

print("State dict:", resultado_load)

model_float.eval()
model_float.to(device)

params = sum(
    p.numel()
    for p in model_float.parameters()
)

print("Parâmetros:", params)

assert params == 6_629_233, (
    f"Número de parâmetros inesperado: {params}"
)


# ============================================================
# CARREGAR EXECUTORCH .PTE
# ============================================================

print()
print("============================================")
print("CARREGANDO MODELO PTE")
print("============================================")

runtime = Runtime.get()

program = runtime.load_program(
    str(PTE_MODEL)
)

pte_method = program.load_method(
    "forward"
)

print("PTE carregado:")
print(PTE_MODEL)


# ============================================================
# LEITURA TIFF
# ============================================================

def read_tif(folder, nome):

    path = folder / f"{nome}.tif"

    with rasterio.open(path) as src:

        return src.read(1).astype(
            np.float32
        )


# ============================================================
# NORMALIZAÇÃO
#
# HyperSTARCOP mag1c + RGB:
#
# mag1c / 1750
# RGB   / 60
# clip [0,2]
#
# ============================================================

def normalizar(x):

    x = x.clone()

    x[:, 0] = torch.clamp(
        x[:, 0] / 1750.0,
        min=0.0,
        max=2.0,
    )

    x[:, 1] = torch.clamp(
        x[:, 1] / 60.0,
        min=0.0,
        max=2.0,
    )

    x[:, 2] = torch.clamp(
        x[:, 2] / 60.0,
        min=0.0,
        max=2.0,
    )

    x[:, 3] = torch.clamp(
        x[:, 3] / 60.0,
        min=0.0,
        max=2.0,
    )

    return x


# ============================================================
# MÉTRICAS
# ============================================================

def confusion(pred, target):

    pred = pred.bool()
    target = target.bool()

    tp = torch.logical_and(
        pred,
        target,
    ).sum().item()

    tn = torch.logical_and(
        ~pred,
        ~target,
    ).sum().item()

    fp = torch.logical_and(
        pred,
        ~target,
    ).sum().item()

    fn = torch.logical_and(
        ~pred,
        target,
    ).sum().item()

    return (
        int(tp),
        int(tn),
        int(fp),
        int(fn),
    )


def calcular_metricas(tp, tn, fp, fn):

    total = (
        tp
        + tn
        + fp
        + fn
    )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * tp
        / (
            2 * tp
            + fp
            + fn
        )
        if (
            2 * tp
            + fp
            + fn
        )
        else 0.0
    )

    iou = (
        tp
        / (
            tp
            + fp
            + fn
        )
        if (
            tp
            + fp
            + fn
        )
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


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
    len(df),
)


# ============================================================
# ACUMULADORES GLOBAIS
# ============================================================

float_tp = 0
float_tn = 0
float_fp = 0
float_fn = 0

pte_tp = 0
pte_tn = 0
pte_fp = 0
pte_fn = 0


max_diff_logits_global = 0.0
soma_diff_logits = 0.0
total_logits = 0

pixels_pred_diferentes_global = 0

resultados = []


# ============================================================
# LOOP
# ============================================================

for index, row in df.iterrows():

    sample_id = row["id"]

    folder = ROOT / sample_id

    print()
    print(
        "============================================"
    )

    print(
        f"IMAGEM {index + 1}/{len(df)}"
    )

    print(
        "ID:",
        sample_id,
    )


    # ========================================================
    # ENTRADA
    # ========================================================

    mag1c = read_tif(
        folder,
        "mag1c",
    )

    red = read_tif(
        folder,
        "TOA_AVIRIS_640nm",
    )

    green = read_tif(
        folder,
        "TOA_AVIRIS_550nm",
    )

    blue = read_tif(
        folder,
        "TOA_AVIRIS_460nm",
    )


    # ========================================================
    # LABEL
    # ========================================================

    label = read_tif(
        folder,
        "labelbinary",
    )

    label_tensor = torch.from_numpy(
        label
    ).bool()

    label_tensor = label_tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )


    # ========================================================
    # STACK
    #
    # [4, 512, 512]
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

    x = x.unsqueeze(0)

    # [1,4,512,512]


    # ========================================================
    # NORMALIZAÇÃO
    # ========================================================

    x_norm = normalizar(
        x
    )

    x_norm = (
        x_norm
        .contiguous()
        .to(device)
    )


    # ========================================================
    # MODELO FLOAT .PTH
    # ========================================================

    with torch.inference_mode():

        logits_float = model_float(
            x_norm
        )


    # ========================================================
    # MODELO EXECUTORCH .PTE
    # ========================================================

    outputs_pte = pte_method.execute(
        [
            x_norm.cpu()
        ]
    )

    logits_pte = outputs_pte[0]


    # ========================================================
    # GARANTIR CPU
    # ========================================================

    logits_float = (
        logits_float
        .detach()
        .cpu()
    )

    logits_pte = (
        logits_pte
        .detach()
        .cpu()
    )


    # ========================================================
    # VERIFICAR SHAPES
    # ========================================================

    if (
        logits_float.shape
        != logits_pte.shape
    ):

        raise RuntimeError(
            f"Shapes diferentes: "
            f"float={logits_float.shape}, "
            f"pte={logits_pte.shape}"
        )


    # ========================================================
    # COMPARAR LOGITS
    # ========================================================

    diff = torch.abs(
        logits_float
        - logits_pte
    )

    max_diff = (
        diff
        .max()
        .item()
    )

    mean_diff = (
        diff
        .mean()
        .item()
    )

    max_diff_logits_global = max(
        max_diff_logits_global,
        max_diff,
    )

    soma_diff_logits += (
        diff.sum().item()
    )

    total_logits += (
        diff.numel()
    )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    prob_float = torch.sigmoid(
        logits_float
    )

    prob_pte = torch.sigmoid(
        logits_pte
    )

    max_diff_prob = torch.abs(
        prob_float
        - prob_pte
    ).max().item()


    # ========================================================
    # SEGMENTAÇÃO
    #
    # sigmoid(logit) > 0.5
    # é equivalente a:
    #
    # logit > 0
    # ========================================================

    pred_float = (
        logits_float >= 0.0
    )

    pred_pte = (
        logits_pte >= 0.0
    )


    # ========================================================
    # DIFERENÇA FLOAT x PTE
    # ========================================================

    pixels_diferentes = (
        pred_float
        != pred_pte
    ).sum().item()

    pixels_pred_diferentes_global += (
        pixels_diferentes
    )


    # ========================================================
    # MÉTRICAS FLOAT
    # ========================================================

    (
        tp_f,
        tn_f,
        fp_f,
        fn_f,
    ) = confusion(
        pred_float,
        label_tensor,
    )

    metricas_float = calcular_metricas(
        tp_f,
        tn_f,
        fp_f,
        fn_f,
    )


    # ========================================================
    # MÉTRICAS PTE
    # ========================================================

    (
        tp_p,
        tn_p,
        fp_p,
        fn_p,
    ) = confusion(
        pred_pte,
        label_tensor,
    )

    metricas_pte = calcular_metricas(
        tp_p,
        tn_p,
        fp_p,
        fn_p,
    )


    # ========================================================
    # ACUMULAR GLOBAL
    # ========================================================

    float_tp += tp_f
    float_tn += tn_f
    float_fp += fp_f
    float_fn += fn_f

    pte_tp += tp_p
    pte_tn += tn_p
    pte_fp += fp_p
    pte_fn += fn_p


    # ========================================================
    # SALVAR RESULTADO INDIVIDUAL
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
                max_diff_prob,

            "pixels_float_pte_diferentes":
                pixels_diferentes,

            "float_accuracy":
                metricas_float[
                    "accuracy"
                ],

            "float_precision":
                metricas_float[
                    "precision"
                ],

            "float_recall":
                metricas_float[
                    "recall"
                ],

            "float_f1":
                metricas_float[
                    "f1"
                ],

            "float_iou":
                metricas_float[
                    "iou"
                ],

            "pte_accuracy":
                metricas_pte[
                    "accuracy"
                ],

            "pte_precision":
                metricas_pte[
                    "precision"
                ],

            "pte_recall":
                metricas_pte[
                    "recall"
                ],

            "pte_f1":
                metricas_pte[
                    "f1"
                ],

            "pte_iou":
                metricas_pte[
                    "iou"
                ],
        }
    )


    # ========================================================
    # RESULTADO IMAGEM
    # ========================================================

    print()

    print(
        "Max diff logits:",
        max_diff,
    )

    print(
        "Mean diff logits:",
        mean_diff,
    )

    print(
        "Max diff prob:",
        max_diff_prob,
    )

    print(
        "Pixels float != PTE:",
        pixels_diferentes,
    )

    print()

    print(
        "FLOAT:"
    )

    print(
        f"  Accuracy:  "
        f"{metricas_float['accuracy']:.6f}"
    )

    print(
        f"  Precision: "
        f"{metricas_float['precision']:.6f}"
    )

    print(
        f"  Recall:    "
        f"{metricas_float['recall']:.6f}"
    )

    print(
        f"  F1:        "
        f"{metricas_float['f1']:.6f}"
    )

    print(
        f"  IoU:       "
        f"{metricas_float['iou']:.6f}"
    )

    print()

    print(
        "PTE:"
    )

    print(
        f"  Accuracy:  "
        f"{metricas_pte['accuracy']:.6f}"
    )

    print(
        f"  Precision: "
        f"{metricas_pte['precision']:.6f}"
    )

    print(
        f"  Recall:    "
        f"{metricas_pte['recall']:.6f}"
    )

    print(
        f"  F1:        "
        f"{metricas_pte['f1']:.6f}"
    )

    print(
        f"  IoU:       "
        f"{metricas_pte['iou']:.6f}"
    )


# ============================================================
# MÉTRICAS GLOBAIS
# ============================================================

metricas_float_global = calcular_metricas(
    float_tp,
    float_tn,
    float_fp,
    float_fn,
)

metricas_pte_global = calcular_metricas(
    pte_tp,
    pte_tn,
    pte_fp,
    pte_fn,
)

mean_diff_logits_global = (
    soma_diff_logits
    / total_logits
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print()
print(
    "================================================"
)
print(
    "RESULTADO GLOBAL"
)
print(
    "================================================"
)

print()
print(
    "Imagens testadas:",
    len(df),
)


print()
print(
    "FLOAT FP32 (.pth)"
)
print(
    "----------------"
)

print(
    "TP:",
    float_tp,
)

print(
    "FP:",
    float_fp,
)

print(
    "FN:",
    float_fn,
)

print(
    "TN:",
    float_tn,
)

print()

print(
    f"Accuracy:  "
    f"{metricas_float_global['accuracy']:.6f}"
)

print(
    f"Precision: "
    f"{metricas_float_global['precision']:.6f}"
)

print(
    f"Recall:    "
    f"{metricas_float_global['recall']:.6f}"
)

print(
    f"F1:        "
    f"{metricas_float_global['f1']:.6f}"
)

print(
    f"IoU:       "
    f"{metricas_float_global['iou']:.6f}"
)


print()
print(
    "EXECUTORCH + XNNPACK (.pte)"
)
print(
    "--------------------------"
)

print(
    "TP:",
    pte_tp,
)

print(
    "FP:",
    pte_fp,
)

print(
    "FN:",
    pte_fn,
)

print(
    "TN:",
    pte_tn,
)

print()

print(
    f"Accuracy:  "
    f"{metricas_pte_global['accuracy']:.6f}"
)

print(
    f"Precision: "
    f"{metricas_pte_global['precision']:.6f}"
)

print(
    f"Recall:    "
    f"{metricas_pte_global['recall']:.6f}"
)

print(
    f"F1:        "
    f"{metricas_pte_global['f1']:.6f}"
)

print(
    f"IoU:       "
    f"{metricas_pte_global['iou']:.6f}"
)


# ============================================================
# DIFERENÇA PTH x PTE
# ============================================================

print()
print(
    "COMPARAÇÃO .PTH x .PTE"
)
print(
    "---------------------"
)

print(
    "Maior diferença logits:",
    max_diff_logits_global,
)

print(
    "Diferença média logits:",
    mean_diff_logits_global,
)

print(
    "Pixels de classificação diferentes:",
    pixels_pred_diferentes_global,
)


# ============================================================
# DELTA DE MÉTRICAS
# ============================================================

print()
print(
    "DELTA PTE - FLOAT"
)
print(
    "-----------------"
)

for nome in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "iou",
]:

    delta = (
        metricas_pte_global[nome]
        -
        metricas_float_global[nome]
    )

    print(
        f"{nome}: {delta:+.8f}"
    )


# ============================================================
# VERIFICAÇÃO NUMÉRICA
# ============================================================

print()
print(
    "VALIDAÇÃO NUMÉRICA"
)
print(
    "------------------"
)

if pixels_pred_diferentes_global == 0:

    print(
        "PTE e FLOAT produzem exatamente "
        "a mesma máscara binária."
    )

else:

    print(
        "ATENÇÃO:",
        pixels_pred_diferentes_global,
        "pixels mudaram de classificação."
    )


# ============================================================
# CSV
# ============================================================

resultados_df = pd.DataFrame(
    resultados
)

output_csv = (
    OUTPUT_DIR
    / "comparacao_float_vs_pte.csv"
)

resultados_df.to_csv(
    output_csv,
    index=False,
)

print()
print(
    "CSV salvo em:",
    output_csv,
)
