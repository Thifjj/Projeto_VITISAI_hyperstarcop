import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import segmentation_models_pytorch as smp

from pytorch_nndct.apis import torch_quantizer


# ============================================================
# ARGUMENTOS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--quant_mode",
    choices=["calib", "test"],
    required=True,
)

parser.add_argument(
    "--calib_samples",
    type=int,
    default=200,
    help="Quantidade de imagens usadas na calibração.",
)

parser.add_argument(
    "--dataset_root",
    type=str,
    default="/workspace/dataset_STARCOP",
    help="Dataset STARCOP completo.",
)

parser.add_argument(
    "--test_root",
    type=str,
    default="STARCOP_mini",
    help="Dataset usado para validação.",
)

parser.add_argument(
    "--deploy",
    action="store_true",
    help="Exporta XModel. Usar com --quant_mode test.",
)

args = parser.parse_args()


if args.deploy and args.quant_mode != "test":
    raise ValueError(
        "--deploy só pode ser usado com --quant_mode test."
    )

if args.calib_samples < 2:
    raise ValueError(
        "--calib_samples precisa ser >= 2."
    )


# ============================================================
# CONFIGURAÇÕES
# ============================================================

SEED = 42
BITWIDTH = 8

TARGET = "DPUCZDX8G_ISA1_B4096"

FULL_ROOT = Path(args.dataset_root)
TEST_ROOT = Path(args.test_root)

TRAIN_CSV = FULL_ROOT / "train.csv"
TEST_CSV = TEST_ROOT / "test_mini10.csv"

WEIGHTS = Path(
    "vitis_ai/float_model/hyperstarcop_network_fp32.pth"
)


# Cada quantidade de calibração tem sua própria pasta.
QUANT_DIR = Path(
    f"vitis_ai/quantized_int8_calib{args.calib_samples}"
)

RESULT_DIR = Path(
    f"vitis_ai/results_int8_calib{args.calib_samples}"
)

QUANT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("====================================================")
print("HYPERSTARCOP - VITIS AI INT8")
print("====================================================")

print("Modo:               ", args.quant_mode)
print("Device:             ", device)
print("Target:             ", TARGET)
print("Bit width:          ", BITWIDTH)
print("Calib samples:      ", args.calib_samples)
print("Dataset calibração: ", FULL_ROOT)
print("Dataset teste:      ", TEST_ROOT)
print("Quant dir:          ", QUANT_DIR)

print()


# ============================================================
# VERIFICAÇÕES
# ============================================================

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"Pesos não encontrados:\n{WEIGHTS}"
    )

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"train.csv não encontrado:\n{TRAIN_CSV}"
    )

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"test_mini10.csv não encontrado:\n{TEST_CSV}"
    )


# ============================================================
# CRIAR MODELO
# ============================================================

def criar_modelo():

    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=4,
        classes=1,
        activation=None,
    )

    state_dict = torch.load(
        WEIGHTS,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# MODELO FP32
# ============================================================

float_model = criar_modelo()

float_model = float_model.to(device)
float_model.eval()


# ============================================================
# MODELO QUE SERÁ QUANTIZADO
# ============================================================

model = criar_modelo()

model = model.to(device)
model.eval()


print("Modelo standalone FP32 carregado.")


# ============================================================
# INPUT PARA CONSTRUÇÃO DO GRAFO
# ============================================================

dummy_input = torch.zeros(
    1,
    4,
    512,
    512,
    dtype=torch.float32,
    device=device,
)


# ============================================================
# QUANTIZER
# ============================================================

print()
print("Criando torch_quantizer...")


quantizer = torch_quantizer(
    quant_mode=args.quant_mode,
    module=model,
    input_args=(dummy_input,),
    output_dir=str(QUANT_DIR),
    bitwidth=BITWIDTH,
    device=device,
    target=TARGET,
)


quant_model = quantizer.quant_model
quant_model.eval()


print("Quantizer criado.")


# ============================================================
# CRIAR ÍNDICE DAS PASTAS DE TREINO
#
# Procura automaticamente:
#
# STARCOP_train_easy
# STARCOP_train_remaining_part1
# ...
# ============================================================

def criar_indice_train():

    print()
    print("====================================================")
    print("INDEXANDO DATASET DE TREINO")
    print("====================================================")

    parts = sorted(
        [
            p
            for p in FULL_ROOT.glob("STARCOP_train*")
            if p.is_dir()
        ]
    )

    if len(parts) == 0:
        raise RuntimeError(
            "Nenhuma pasta STARCOP_train* encontrada."
        )


    print("Partes encontradas:")

    for part in parts:
        print("  ", part.name)


    indice = {}

    for part in parts:

        contador = 0

        for folder in part.iterdir():

            if not folder.is_dir():
                continue

            indice[folder.name] = folder

            contador += 1

        print(
            f"{part.name}: {contador} diretórios"
        )


    print()
    print(
        "Total de IDs indexados:",
        len(indice)
    )


    return indice


# ============================================================
# LEITURA TIFF
# ============================================================

def read_tif(folder, nome):

    path = folder / f"{nome}.tif"

    if not path.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{path}"
        )


    with rasterio.open(path) as src:

        array = src.read(1).astype(
            np.float32
        )


    return array


# ============================================================
# NORMALIZAÇÃO
#
# Exatamente igual ao modelo original:
#
# mag1c / 1750
# RGB   / 60
# clip [0,2]
# ============================================================

def normalizar(x):

    x = x.clone()


    x[:, 0] = torch.clamp(
        x[:, 0] / 1750.0,
        0.0,
        2.0,
    )


    x[:, 1] = torch.clamp(
        x[:, 1] / 60.0,
        0.0,
        2.0,
    )


    x[:, 2] = torch.clamp(
        x[:, 2] / 60.0,
        0.0,
        2.0,
    )


    x[:, 3] = torch.clamp(
        x[:, 3] / 60.0,
        0.0,
        2.0,
    )


    return x


# ============================================================
# CARREGAR INPUT
# ============================================================

def carregar_input(folder):

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


    shapes = {
        mag1c.shape,
        red.shape,
        green.shape,
        blue.shape,
    }


    if len(shapes) != 1:

        raise RuntimeError(
            f"Canais com shapes diferentes em:\n{folder}\n"
            f"Shapes: {shapes}"
        )


    # Nosso modelo e quantização atuais foram preparados
    # para 512x512.
    if mag1c.shape != (512, 512):

        raise RuntimeError(
            f"Imagem não é 512x512:\n"
            f"{folder}\n"
            f"Shape encontrado: {mag1c.shape}"
        )


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


    # [4,512,512]
    # ->
    # [1,4,512,512]

    x = x.unsqueeze(0)

    x = x.to(device)

    x = normalizar(x)

    return x


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def matriz_confusao(
    pred,
    label,
):

    TP = int(
        np.sum(
            (pred == 1)
            &
            (label == 1)
        )
    )

    FP = int(
        np.sum(
            (pred == 1)
            &
            (label == 0)
        )
    )

    FN = int(
        np.sum(
            (pred == 0)
            &
            (label == 1)
        )
    )

    TN = int(
        np.sum(
            (pred == 0)
            &
            (label == 0)
        )
    )


    return TP, FP, FN, TN


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    TP,
    FP,
    FN,
    TN,
):

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    iou = (
        TP / (TP + FP + FN)
        if TP + FP + FN > 0
        else 0.0
    )

    total = (
        TP +
        FP +
        FN +
        TN
    )

    accuracy = (
        (TP + TN) / total
        if total > 0
        else 0.0
    )


    return (
        precision,
        recall,
        f1,
        iou,
        accuracy,
    )


# ============================================================
# SELECIONAR CALIBRAÇÃO BALANCEADA
# ============================================================

def selecionar_calibracao():

    df = pd.read_csv(
        TRAIN_CSV
    )


    print()
    print("====================================================")
    print("TRAIN.CSV")
    print("====================================================")

    print(
        "Total:",
        len(df)
    )

    print()

    print(
        df["has_plume"].value_counts()
    )


    # ========================================================
    # Divisão aproximadamente 50/50
    # ========================================================

    n_positive = (
        args.calib_samples // 2
    )

    n_negative = (
        args.calib_samples -
        n_positive
    )


    positive = df[
        df["has_plume"] == True
    ]


    negative = df[
        df["has_plume"] == False
    ]


    if len(positive) < n_positive:

        raise RuntimeError(
            "Não existem amostras positivas suficientes."
        )


    if len(negative) < n_negative:

        raise RuntimeError(
            "Não existem amostras negativas suficientes."
        )


    positive = positive.sample(
        n=n_positive,
        random_state=SEED,
    )


    negative = negative.sample(
        n=n_negative,
        random_state=SEED,
    )


    calib_df = pd.concat(
        [
            positive,
            negative,
        ]
    )


    # Embaralhar mantendo reprodutibilidade

    calib_df = calib_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(
        drop=True
    )


    print()
    print("====================================================")
    print("AMOSTRAS SELECIONADAS PARA CALIBRAÇÃO")
    print("====================================================")

    print(
        "Total:",
        len(calib_df)
    )

    print(
        "Com pluma:",
        int(calib_df["has_plume"].sum())
    )

    print(
        "Sem pluma:",
        int(
            (~calib_df["has_plume"]).sum()
        )
    )


    # Salvar exatamente quais imagens foram usadas.
    output_csv = (
        RESULT_DIR /
        f"calibration_samples_{args.calib_samples}.csv"
    )

    calib_df.to_csv(
        output_csv,
        index=False,
    )


    print()
    print(
        "Lista salva em:"
    )

    print(
        output_csv
    )


    return calib_df


# ============================================================
# CALIBRAÇÃO
# ============================================================

if args.quant_mode == "calib":

    indice_train = criar_indice_train()

    calib_df = selecionar_calibracao()


    # ========================================================
    # VERIFICAR SE TODOS OS IDs EXISTEM
    # ========================================================

    faltando = []

    for sample_id in calib_df["id"]:

        if sample_id not in indice_train:

            faltando.append(
                sample_id
            )


    if faltando:

        print()
        print("IDs não encontrados:")

        for sample_id in faltando:
            print(sample_id)

        raise RuntimeError(
            f"{len(faltando)} amostras não encontradas."
        )


    # ========================================================
    # CALIBRAÇÃO
    # ========================================================

    print()
    print("====================================================")
    print("CALIBRAÇÃO INT8")
    print("====================================================")

    print(
        "Total:",
        len(calib_df)
    )

    print()


    with torch.no_grad():

        for index, row in calib_df.iterrows():

            sample_id = row["id"]

            folder = indice_train[
                sample_id
            ]


            print(
                f"[{index + 1:03d}/"
                f"{len(calib_df):03d}] "
                f"plume={row['has_plume']} "
                f"{sample_id}"
            )


            x = carregar_input(
                folder
            )


            _ = quant_model(
                x
            )


    # ========================================================
    # EXPORTAR CONFIGURAÇÃO
    # ========================================================

    print()
    print(
        "Exportando configuração INT8..."
    )


    quantizer.export_quant_config()


    print()
    print("====================================================")
    print("CALIBRAÇÃO FINALIZADA")
    print("====================================================")

    print()

    print(
        "Configuração salva em:"
    )

    print(
        QUANT_DIR
    )

    print()

    print("Agora rode:")

    print()

    print(
        f"python vitis_ai/scripts/05_quantizar_int8.py "
        f"--quant_mode test "
        f"--calib_samples {args.calib_samples}"
    )


# ============================================================
# TEST
# ============================================================

elif args.quant_mode == "test":

    df = pd.read_csv(
        TEST_CSV
    )


    if args.deploy:

        df = df.iloc[:1].copy()


    print()
    print("====================================================")
    print("VALIDAÇÃO FP32 x INT8")
    print("====================================================")

    print(
        "Imagens:",
        len(df)
    )


    # ========================================================
    # ACUMULADORES
    # ========================================================

    TP_FP32_GLOBAL = 0
    FP_FP32_GLOBAL = 0
    FN_FP32_GLOBAL = 0
    TN_FP32_GLOBAL = 0

    TP_INT8_GLOBAL = 0
    FP_INT8_GLOBAL = 0
    FN_INT8_GLOBAL = 0
    TN_INT8_GLOBAL = 0

    pixels_diferentes_global = 0

    max_diff_logits_global = 0.0
    max_diff_prob_global = 0.0

    soma_diff_logits = 0.0
    total_logits = 0

    resultados = []


    # ========================================================
    # LOOP TESTE
    # ========================================================

    for pos, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        sample_id = row["id"]

        folder = (
            TEST_ROOT /
            sample_id
        )


        print()
        print(
            "===================================================="
        )

        print(
            f"[{pos}/{len(df)}] "
            f"{sample_id}"
        )

        print(
            "===================================================="
        )


        # ====================================================
        # INPUT
        # ====================================================

        x = carregar_input(
            folder
        )


        # ====================================================
        # GROUND TRUTH
        # ====================================================

        label = read_tif(
            folder,
            "labelbinary",
        )


        label_np = (
            label > 0
        ).astype(
            np.uint8
        )


        # ====================================================
        # FP32
        # ====================================================

        with torch.no_grad():

            logits_fp32 = float_model(
                x
            )


        # ====================================================
        # INT8
        # ====================================================

        with torch.no_grad():

            logits_int8 = quant_model(
                x
            )


        # ====================================================
        # SIGMOID
        # ====================================================

        prob_fp32 = torch.sigmoid(
            logits_fp32
        )

        prob_int8 = torch.sigmoid(
            logits_int8
        )


        # ====================================================
        # THRESHOLD
        # ====================================================

        pred_fp32 = (
            prob_fp32 > 0.5
        ).to(
            torch.uint8
        )


        pred_int8 = (
            prob_int8 > 0.5
        ).to(
            torch.uint8
        )


        pred_fp32_np = (
            pred_fp32[0, 0]
            .cpu()
            .numpy()
        )


        pred_int8_np = (
            pred_int8[0, 0]
            .cpu()
            .numpy()
        )


        # ====================================================
        # MATRIZ FP32
        # ====================================================

        (
            TP_FP32,
            FP_FP32,
            FN_FP32,
            TN_FP32,
        ) = matriz_confusao(
            pred_fp32_np,
            label_np,
        )


        TP_FP32_GLOBAL += TP_FP32
        FP_FP32_GLOBAL += FP_FP32
        FN_FP32_GLOBAL += FN_FP32
        TN_FP32_GLOBAL += TN_FP32


        # ====================================================
        # MATRIZ INT8
        # ====================================================

        (
            TP_INT8,
            FP_INT8,
            FN_INT8,
            TN_INT8,
        ) = matriz_confusao(
            pred_int8_np,
            label_np,
        )


        TP_INT8_GLOBAL += TP_INT8
        FP_INT8_GLOBAL += FP_INT8
        FN_INT8_GLOBAL += FN_INT8
        TN_INT8_GLOBAL += TN_INT8


        # ====================================================
        # MÉTRICAS
        # ====================================================

        metrics_fp32 = calcular_metricas(
            TP_FP32,
            FP_FP32,
            FN_FP32,
            TN_FP32,
        )


        metrics_int8 = calcular_metricas(
            TP_INT8,
            FP_INT8,
            FN_INT8,
            TN_INT8,
        )


        (
            precision_fp32,
            recall_fp32,
            f1_fp32,
            iou_fp32,
            accuracy_fp32,
        ) = metrics_fp32


        (
            precision_int8,
            recall_int8,
            f1_int8,
            iou_int8,
            accuracy_int8,
        ) = metrics_int8


        # ====================================================
        # DIFERENÇAS
        # ====================================================

        diff_logits = torch.abs(
            logits_fp32 -
            logits_int8
        )


        max_diff_logits = (
            diff_logits.max().item()
        )


        mean_diff_logits = (
            diff_logits.mean().item()
        )


        diff_prob = torch.abs(
            prob_fp32 -
            prob_int8
        )


        max_diff_prob = (
            diff_prob.max().item()
        )


        pixels_diferentes = int(
            (
                pred_fp32 !=
                pred_int8
            ).sum().item()
        )


        pixels_diferentes_global += (
            pixels_diferentes
        )


        max_diff_logits_global = max(
            max_diff_logits_global,
            max_diff_logits,
        )


        max_diff_prob_global = max(
            max_diff_prob_global,
            max_diff_prob,
        )


        soma_diff_logits += (
            diff_logits.sum().item()
        )

        total_logits += (
            diff_logits.numel()
        )


        # ====================================================
        # SALVAR
        # ====================================================

        resultados.append(
            {
                "id":
                    sample_id,

                "TP_fp32":
                    TP_FP32,

                "FP_fp32":
                    FP_FP32,

                "FN_fp32":
                    FN_FP32,

                "TN_fp32":
                    TN_FP32,

                "precision_fp32":
                    precision_fp32,

                "recall_fp32":
                    recall_fp32,

                "f1_fp32":
                    f1_fp32,

                "iou_fp32":
                    iou_fp32,

                "TP_int8":
                    TP_INT8,

                "FP_int8":
                    FP_INT8,

                "FN_int8":
                    FN_INT8,

                "TN_int8":
                    TN_INT8,

                "precision_int8":
                    precision_int8,

                "recall_int8":
                    recall_int8,

                "f1_int8":
                    f1_int8,

                "iou_int8":
                    iou_int8,

                "pixels_fp32_int8_diferentes":
                    pixels_diferentes,

                "max_diff_logits":
                    max_diff_logits,

                "mean_diff_logits":
                    mean_diff_logits,

                "max_diff_prob":
                    max_diff_prob,
            }
        )


        print()
        print("FP32")

        print(
            f"Precision: {precision_fp32:.4f}"
        )

        print(
            f"Recall:    {recall_fp32:.4f}"
        )

        print(
            f"F1:        {f1_fp32:.4f}"
        )

        print(
            f"IoU:       {iou_fp32:.4f}"
        )


        print()
        print("INT8")

        print(
            f"Precision: {precision_int8:.4f}"
        )

        print(
            f"Recall:    {recall_int8:.4f}"
        )

        print(
            f"F1:        {f1_int8:.4f}"
        )

        print(
            f"IoU:       {iou_int8:.4f}"
        )


        print()

        print(
            "Pixels FP32 != INT8:",
            pixels_diferentes,
        )


    # ========================================================
    # DEPLOY
    # ========================================================

    if args.deploy:

        print()
        print("====================================================")
        print("EXPORTANDO XMODEL")
        print("====================================================")


        quantizer.export_xmodel(
            output_dir=str(
                QUANT_DIR
            ),
            deploy_check=False,
        )


        print()
        print(
            "XModel salvo em:"
        )

        print(
            QUANT_DIR
        )


    # ========================================================
    # MÉTRICAS GLOBAIS
    # ========================================================

    else:

        metrics_fp32_global = calcular_metricas(
            TP_FP32_GLOBAL,
            FP_FP32_GLOBAL,
            FN_FP32_GLOBAL,
            TN_FP32_GLOBAL,
        )


        metrics_int8_global = calcular_metricas(
            TP_INT8_GLOBAL,
            FP_INT8_GLOBAL,
            FN_INT8_GLOBAL,
            TN_INT8_GLOBAL,
        )


        (
            precision_fp32_global,
            recall_fp32_global,
            f1_fp32_global,
            iou_fp32_global,
            accuracy_fp32_global,
        ) = metrics_fp32_global


        (
            precision_int8_global,
            recall_int8_global,
            f1_int8_global,
            iou_int8_global,
            accuracy_int8_global,
        ) = metrics_int8_global


        mean_diff_logits_global = (
            soma_diff_logits /
            total_logits
        )


        # ====================================================
        # FP32
        # ====================================================

        print()
        print()
        print("====================================================")
        print("FP32 - GLOBAL")
        print("====================================================")

        print(
            "TP:",
            TP_FP32_GLOBAL,
        )

        print(
            "FP:",
            FP_FP32_GLOBAL,
        )

        print(
            "FN:",
            FN_FP32_GLOBAL,
        )

        print(
            "TN:",
            TN_FP32_GLOBAL,
        )

        print()

        print(
            f"Precision: {precision_fp32_global:.4f}"
        )

        print(
            f"Recall:    {recall_fp32_global:.4f}"
        )

        print(
            f"F1:        {f1_fp32_global:.4f}"
        )

        print(
            f"IoU:       {iou_fp32_global:.4f}"
        )


        # ====================================================
        # INT8
        # ====================================================

        print()
        print("====================================================")
        print("INT8 - GLOBAL")
        print("====================================================")

        print(
            "TP:",
            TP_INT8_GLOBAL,
        )

        print(
            "FP:",
            FP_INT8_GLOBAL,
        )

        print(
            "FN:",
            FN_INT8_GLOBAL,
        )

        print(
            "TN:",
            TN_INT8_GLOBAL,
        )

        print()

        print(
            f"Precision: {precision_int8_global:.4f}"
        )

        print(
            f"Recall:    {recall_int8_global:.4f}"
        )

        print(
            f"F1:        {f1_int8_global:.4f}"
        )

        print(
            f"IoU:       {iou_int8_global:.4f}"
        )


        # ====================================================
        # DELTAS
        # ====================================================

        delta_precision = (
            precision_int8_global -
            precision_fp32_global
        )

        delta_recall = (
            recall_int8_global -
            recall_fp32_global
        )

        delta_f1 = (
            f1_int8_global -
            f1_fp32_global
        )

        delta_iou = (
            iou_int8_global -
            iou_fp32_global
        )


        print()
        print("====================================================")
        print("INT8 - FP32")
        print("====================================================")

        print(
            f"Δ Precision: {delta_precision:+.4f}"
        )

        print(
            f"Δ Recall:    {delta_recall:+.4f}"
        )

        print(
            f"Δ F1:        {delta_f1:+.4f}"
        )

        print(
            f"Δ IoU:       {delta_iou:+.4f}"
        )

        print()

        print(
            "Pixels diferentes:",
            pixels_diferentes_global,
        )

        print(
            "Max diff logits:",
            max_diff_logits_global,
        )

        print(
            "Mean diff logits:",
            mean_diff_logits_global,
        )

        print(
            "Max diff prob:",
            max_diff_prob_global,
        )


        # ====================================================
        # CSV INDIVIDUAL
        # ====================================================

        individual_df = pd.DataFrame(
            resultados
        )

        individual_path = (
            RESULT_DIR /
            "comparacao_fp32_int8_por_imagem.csv"
        )

        individual_df.to_csv(
            individual_path,
            index=False,
        )


        # ====================================================
        # CSV GLOBAL
        # ====================================================

        global_df = pd.DataFrame(
            [
                {
                    "calib_samples":
                        args.calib_samples,

                    "test_images":
                        len(df),

                    "precision_fp32":
                        precision_fp32_global,

                    "recall_fp32":
                        recall_fp32_global,

                    "f1_fp32":
                        f1_fp32_global,

                    "iou_fp32":
                        iou_fp32_global,

                    "precision_int8":
                        precision_int8_global,

                    "recall_int8":
                        recall_int8_global,

                    "f1_int8":
                        f1_int8_global,

                    "iou_int8":
                        iou_int8_global,

                    "delta_precision":
                        delta_precision,

                    "delta_recall":
                        delta_recall,

                    "delta_f1":
                        delta_f1,

                    "delta_iou":
                        delta_iou,

                    "pixels_diferentes":
                        pixels_diferentes_global,

                    "max_diff_logits":
                        max_diff_logits_global,

                    "mean_diff_logits":
                        mean_diff_logits_global,

                    "max_diff_prob":
                        max_diff_prob_global,
                }
            ]
        )


        global_path = (
            RESULT_DIR /
            "comparacao_fp32_int8_global.csv"
        )


        global_df.to_csv(
            global_path,
            index=False,
        )


        print()
        print("====================================================")
        print("RESULTADOS SALVOS")
        print("====================================================")

        print(
            individual_path
        )

        print(
            global_path
        )