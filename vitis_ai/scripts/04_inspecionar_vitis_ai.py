import torch
import segmentation_models_pytorch as smp

from pathlib import Path
from pytorch_nndct.apis import Inspector


# ============================================================
# CAMINHOS
# ============================================================

WEIGHTS = Path(
    "vitis_ai/float_model/hyperstarcop_network_fp32.pth"
)

OUTPUT_DIR = Path(
    "vitis_ai/inspect"
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
# RECRIAR MODELO PURO
# ============================================================

model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=4,
    classes=1,
    activation=None,
)


# ============================================================
# CARREGAR PESOS
# ============================================================

state_dict = torch.load(
    WEIGHTS,
    map_location="cpu"
)

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(device)

model.eval()


print("Modelo carregado.")
print(type(model))


# ============================================================
# INPUT DO HYPERSTARCOP
#
# Já normalizado:
#
# [B,C,H,W]
# [1,4,512,512]
# ============================================================

dummy_input = torch.randn(
    1,
    4,
    512,
    512,
    device=device
)


print(
    "Input:",
    dummy_input.shape
)


# ============================================================
# TARGET DA ZCU104
#
# DPUCZDX8G B4096
# ============================================================

TARGET = "DPUCZDX8G_ISA1_B4096"


print(
    "Target:",
    TARGET
)


# ============================================================
# MODEL INSPECTOR
# ============================================================

inspector = Inspector(
    TARGET
)


inspector.inspect(
    model,
    (dummy_input,),
    device=device,
    output_dir=str(OUTPUT_DIR),
    verbose_level=2,
    image_format="svg",
)


print()
print("===================================")
print("INSPEÇÃO FINALIZADA")
print("===================================")

print(
    "Resultados:",
    OUTPUT_DIR
)