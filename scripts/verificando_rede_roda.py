import torch
import omegaconf
import ast

from starcop.models.model_module import ModelModule

# ============================================
# ARQUIVOS
# ============================================

CHECKPOINT = "model/final_checkpoint_model.ckpt"
CONFIG_CHECKPOINT = "model/config.yaml"

# Configuração base instalada pelo STARCOP
import starcop
import os

STARCOP_DIR = os.path.dirname(starcop.__file__)
CONFIG_BASE = os.path.join(STARCOP_DIR, "config.yaml")


# ============================================
# CONFIG
# ============================================

config_general = omegaconf.OmegaConf.load(CONFIG_BASE)
config_model = omegaconf.OmegaConf.load(CONFIG_CHECKPOINT)

config = omegaconf.OmegaConf.merge(
    config_general,
    config_model
)


# O config salvo no HuggingFace possui parte da
# configuração dentro do metadata do wandb
dataset_dict = ast.literal_eval(
    config_model["_content"]["value"]["dataset"]
)

config.dataset = dataset_dict


# ============================================
# MODELO
# ============================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ModelModule.load_from_checkpoint(
    CHECKPOINT,
    settings=config,
    map_location=device,
    weights_only=False
)

model = model.to(device)
model.eval()


print("\n===== CHECKPOINT CARREGADO =====")
print("Device:", device)
print("Número de canais:", model.num_channels)

#rint("\nRede:")
#print(model.network)
# ============================================================
# TESTE SIMPLES DE INFERÊNCIA
# ============================================================

x = torch.zeros(
    (1, 4, 512, 512),
    dtype=torch.float32,
    device=device
)

print("\nEntrada:", x.shape)

with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).to(torch.uint8)

print("Logits :", logits.shape)
print("Prob   :", prob.shape)
print("Pred   :", pred.shape)

print("\nProbabilidade mínima:", prob.min().item())
print("Probabilidade máxima:", prob.max().item())
print("Pixels classificados como metano:", pred.sum().item())