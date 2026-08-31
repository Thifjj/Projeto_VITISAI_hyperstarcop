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

print("\nRede:")
print(model.network)