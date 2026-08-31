#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# HyperSTARCOP / STARCOP - Preparação completa do ambiente
# ============================================================
#
# Uso:
#
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# Execute este script na raiz do projeto:
#
#   Projeto_Metano/
#
# O script cria/prepara:
#
#   venv/
#   STARCOP/
#   model/
#   STARCOP_mini/
#   resultados/
#   resultados_dataset/
#
# ============================================================

PROJECT_ROOT="$(pwd)"

VENV_DIR="$PROJECT_ROOT/venv"
MODEL_DIR="$PROJECT_ROOT/model"
RESULT_DIR="$PROJECT_ROOT/resultados"
RESULT_DATASET_DIR="$PROJECT_ROOT/resultados_dataset"

STARCOP_DIR="$PROJECT_ROOT/STARCOP"

HF_REPO="isp-uv-es/starcop"
HF_CONFIG="models/hyperstarcop_mag1c_rgb/config.yaml"
HF_CHECKPOINT="models/hyperstarcop_mag1c_rgb/final_checkpoint_model.ckpt"

STARCOP_GIT="https://github.com/spaceml-org/STARCOP.git"

STARCOP_MINI_ZIP="$PROJECT_ROOT/STARCOP_mini.zip"
STARCOP_MINI_DIR="$PROJECT_ROOT/STARCOP_mini"

# ID usado pelo STARCOP_mini oficial
STARCOP_MINI_GDRIVE_ID="1Qw96Drmk2jzBYSED0YPEUyuc2DnBechl"


# ============================================================
# CORES
# ============================================================

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m"


info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[AVISO]${NC} $1"
}

fail() {
    echo -e "${RED}[ERRO]${NC} $1"
    exit 1
}


# ============================================================
# TRATAMENTO DE ERROS
# ============================================================

trap 'echo -e "\n${RED}[ERRO]${NC} Falha na linha $LINENO."' ERR


# ============================================================
# VERIFICAR DIRETÓRIO
# ============================================================

echo
echo "============================================================"
echo " PREPARAÇÃO DO AMBIENTE - HyperSTARCOP"
echo "============================================================"
echo
echo "Projeto:"
echo "  $PROJECT_ROOT"
echo


# ============================================================
# DEPENDÊNCIAS DO SISTEMA
# ============================================================

info "Verificando dependências do sistema..."

SYSTEM_MISSING=()

command -v python3 >/dev/null 2>&1 || SYSTEM_MISSING+=("python3")
command -v git >/dev/null 2>&1 || SYSTEM_MISSING+=("git")
command -v unzip >/dev/null 2>&1 || SYSTEM_MISSING+=("unzip")

if [ ${#SYSTEM_MISSING[@]} -gt 0 ]; then

    warn "Dependências ausentes: ${SYSTEM_MISSING[*]}"

    if command -v apt >/dev/null 2>&1; then

        info "Tentando instalar dependências via apt..."

        sudo apt update

        sudo apt install -y \
            python3 \
            python3-venv \
            python3-pip \
            python3-dev \
            git \
            unzip \
            build-essential \
            libgdal-dev \
            gdal-bin

    else
        fail "Instale manualmente: ${SYSTEM_MISSING[*]}"
    fi

else
    ok "Dependências básicas encontradas."
fi


# python3-venv pode estar ausente mesmo com python3 instalado
if ! python3 -m venv --help >/dev/null 2>&1; then

    if command -v apt >/dev/null 2>&1; then
        info "Instalando python3-venv..."
        sudo apt install -y python3-venv
    else
        fail "O módulo python3-venv não está disponível."
    fi

fi


# ============================================================
# MOSTRAR VERSÃO DO PYTHON
# ============================================================

PYTHON_VERSION="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

info "Python encontrado: $PYTHON_VERSION"


# ============================================================
# CRIAR DIRETÓRIOS
# ============================================================

info "Criando diretórios do projeto..."

mkdir -p "$MODEL_DIR"
mkdir -p "$RESULT_DIR"
mkdir -p "$RESULT_DATASET_DIR"

ok "Diretórios preparados."


# ============================================================
# CRIAR VENV
# ============================================================

if [ ! -d "$VENV_DIR" ]; then

    info "Criando ambiente virtual em:"
    echo "  $VENV_DIR"

    python3 -m venv "$VENV_DIR"

    ok "venv criado."

else
    ok "venv já existe."
fi


# ============================================================
# ATIVAR VENV
# ============================================================

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

ok "venv ativado."

echo
echo "Python do ambiente:"
which python3

echo
python3 --version


# ============================================================
# ATUALIZAR FERRAMENTAS DE BUILD
# ============================================================

info "Atualizando pip/setuptools/wheel..."

python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel

ok "Ferramentas atualizadas."


# ============================================================
# DEPENDÊNCIAS PYTHON
# ============================================================

info "Instalando dependências Python..."

#
# pytorch-lightning 2.6.5 é a versão observada durante
# a validação realizada neste projeto.
#
# torch/torchvision ficam sem pin rígido para permitir
# instalação da build adequada ao sistema.
#

python3 -m pip install \
    torch \
    torchvision \
    "pytorch-lightning==2.6.5" \
    omegaconf \
    hydra-core \
    wandb \
    segmentation-models-pytorch \
    torchmetrics \
    kornia \
    fsspec \
    rasterio \
    pandas \
    numpy \
    matplotlib \
    tqdm \
    scikit-learn \
    gdown \
    huggingface_hub

ok "Dependências Python instaladas."


# ============================================================
# CLONAR STARCOP
# ============================================================

info "Verificando código-fonte do STARCOP..."

STARCOP_SOURCE=""

# Estrutura limpa:
#
# STARCOP/starcop/
#
if [ -f "$STARCOP_DIR/starcop/models/model_module.py" ]; then

    STARCOP_SOURCE="$STARCOP_DIR"

    ok "STARCOP encontrado em:"
    echo "  $STARCOP_SOURCE"

# Estrutura usada durante os testes anteriores:
#
# STARCOP/STARCOP/starcop/
#
elif [ -f "$STARCOP_DIR/STARCOP/starcop/models/model_module.py" ]; then

    STARCOP_SOURCE="$STARCOP_DIR/STARCOP"

    ok "STARCOP encontrado em:"
    echo "  $STARCOP_SOURCE"

else

    if [ -d "$STARCOP_DIR" ] && [ "$(find "$STARCOP_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then

        warn "A pasta STARCOP existe, mas não contém uma instalação reconhecida."
        warn "Tentando clonar dentro de STARCOP/STARCOP..."

        git clone "$STARCOP_GIT" "$STARCOP_DIR/STARCOP"

        STARCOP_SOURCE="$STARCOP_DIR/STARCOP"

    else

        info "Clonando STARCOP oficial..."

        rm -rf "$STARCOP_DIR"

        git clone "$STARCOP_GIT" "$STARCOP_DIR"

        STARCOP_SOURCE="$STARCOP_DIR"

    fi

fi


if [ ! -f "$STARCOP_SOURCE/starcop/models/model_module.py" ]; then
    fail "model_module.py não foi encontrado após o clone."
fi

ok "Código STARCOP disponível."


# ============================================================
# PYTHONPATH
# ============================================================

export PYTHONPATH="$STARCOP_SOURCE:${PYTHONPATH:-}"

ok "PYTHONPATH configurado para esta execução:"
echo "  $STARCOP_SOURCE"


# ============================================================
# CRIAR SCRIPT DE ATIVAÇÃO DO PROJETO
# ============================================================

ACTIVATE_PROJECT="$PROJECT_ROOT/activate_project.sh"

cat > "$ACTIVATE_PROJECT" <<EOF
#!/usr/bin/env bash

PROJECT_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

source "\$PROJECT_ROOT/venv/bin/activate"

export PYTHONPATH="$STARCOP_SOURCE:\${PYTHONPATH:-}"

echo "Ambiente Projeto_Metano ativado."
echo "Python: \$(which python3)"
echo "PYTHONPATH: $STARCOP_SOURCE"
EOF

chmod +x "$ACTIVATE_PROJECT"

ok "Criado:"
echo "  activate_project.sh"


# ============================================================
# DOWNLOAD DO MODELO - HUGGING FACE
# ============================================================

CHECKPOINT_PATH="$MODEL_DIR/final_checkpoint_model.ckpt"
CONFIG_PATH="$MODEL_DIR/config.yaml"

if [ -f "$CHECKPOINT_PATH" ] && [ -s "$CHECKPOINT_PATH" ]; then

    CHECKPOINT_SIZE="$(stat -c%s "$CHECKPOINT_PATH")"

    ok "Checkpoint já existe:"
    echo "  $CHECKPOINT_PATH"
    echo "  $CHECKPOINT_SIZE bytes"

else

    info "Baixando checkpoint oficial do Hugging Face..."

    python3 - <<PY
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

repo = "$HF_REPO"

src = hf_hub_download(
    repo_id=repo,
    filename="$HF_CHECKPOINT",
)

dst = Path(r"$CHECKPOINT_PATH")
dst.parent.mkdir(parents=True, exist_ok=True)

shutil.copy2(src, dst)

print("Checkpoint salvo em:", dst)
PY

fi


if [ -f "$CONFIG_PATH" ] && [ -s "$CONFIG_PATH" ]; then

    ok "Config do modelo já existe:"
    echo "  $CONFIG_PATH"

else

    info "Baixando config oficial do Hugging Face..."

    python3 - <<PY
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

repo = "$HF_REPO"

src = hf_hub_download(
    repo_id=repo,
    filename="$HF_CONFIG",
)

dst = Path(r"$CONFIG_PATH")
dst.parent.mkdir(parents=True, exist_ok=True)

shutil.copy2(src, dst)

print("Config salvo em:", dst)
PY

fi


# ============================================================
# VERIFICAR CHECKPOINT
# ============================================================

if [ ! -s "$CHECKPOINT_PATH" ]; then
    fail "Checkpoint não encontrado ou vazio."
fi

CHECKPOINT_SIZE="$(stat -c%s "$CHECKPOINT_PATH")"

ok "Checkpoint disponível:"
echo "  $CHECKPOINT_PATH"
echo "  $CHECKPOINT_SIZE bytes"


# ============================================================
# DOWNLOAD DO STARCOP MINI
# ============================================================

if [ -d "$STARCOP_MINI_DIR" ] && \
   [ -f "$STARCOP_MINI_DIR/test_mini10.csv" ]; then

    ok "STARCOP_mini já existe."

else

    if [ ! -f "$STARCOP_MINI_ZIP" ]; then

        info "Baixando STARCOP_mini..."

        python3 - <<PY
import gdown

gdown.download(
    id="$STARCOP_MINI_GDRIVE_ID",
    output=r"$STARCOP_MINI_ZIP",
    quiet=False
)
PY

    else
        ok "STARCOP_mini.zip já existe."
    fi

    info "Extraindo STARCOP_mini..."

    unzip -q -o "$STARCOP_MINI_ZIP" -d "$PROJECT_ROOT"

fi


if [ ! -f "$STARCOP_MINI_DIR/test_mini10.csv" ]; then

    warn "test_mini10.csv não encontrado diretamente em STARCOP_mini."

    FOUND_MINI_CSV="$(find "$PROJECT_ROOT" -maxdepth 3 -name test_mini10.csv -print -quit || true)"

    if [ -n "$FOUND_MINI_CSV" ]; then
        warn "Foi encontrado em:"
        echo "  $FOUND_MINI_CSV"
    else
        fail "Não foi possível localizar test_mini10.csv."
    fi

else
    ok "STARCOP_mini preparado."
fi


# ============================================================
# VERIFICAR TORCH_FORCE_WEIGHTS_ONLY_LOAD
# ============================================================

if [ "${TORCH_FORCE_WEIGHTS_ONLY_LOAD:-}" = "1" ] || \
   [ "${TORCH_FORCE_WEIGHTS_ONLY_LOAD:-}" = "true" ] || \
   [ "${TORCH_FORCE_WEIGHTS_ONLY_LOAD:-}" = "yes" ]; then

    warn "TORCH_FORCE_WEIGHTS_ONLY_LOAD está ativo."
    warn "Isso impede o carregamento do checkpoint antigo."

    unset TORCH_FORCE_WEIGHTS_ONLY_LOAD

    ok "Variável removida para esta sessão."
fi


# ============================================================
# TESTAR IMPORTS
# ============================================================

info "Testando imports..."

python3 - <<'PY'
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import rasterio
import pandas
import numpy
import matplotlib
import omegaconf
import starcop

from starcop.models.model_module import ModelModule

print()
print("===== VERSÕES =====")
print("torch:", torch.__version__)
print("pytorch_lightning:", pl.__version__)
print("segmentation_models_pytorch:", smp.__version__)
print("starcop:", starcop.__file__)

print()
print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("ModelModule: OK")
PY

ok "Imports principais funcionando."


# ============================================================
# TESTE DO CHECKPOINT
# ============================================================

info "Testando carregamento do checkpoint..."

python3 - <<PY
import os
import torch
from omegaconf import OmegaConf

import starcop
from starcop.models.model_module import ModelModule


checkpoint = r"$CHECKPOINT_PATH"

starcop_dir = os.path.dirname(starcop.__file__)
config_base = os.path.join(starcop_dir, "config.yaml")

config = OmegaConf.load(config_base)

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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ModelModule.load_from_checkpoint(
    checkpoint,
    settings=config,
    map_location=device,
    weights_only=False,
)

model = model.to(device)
model.eval()

print()
print("===== CHECKPOINT =====")
print("Checkpoint: OK")
print("Device:", device)
print("Número de canais:", model.num_channels)
PY

ok "Checkpoint carregado corretamente."


# ============================================================
# SMOKE TEST 512x512
# ============================================================

info "Executando smoke test [1,4,512,512]..."

python3 - <<PY
import os
import torch
from omegaconf import OmegaConf

import starcop
from starcop.models.model_module import ModelModule


checkpoint = r"$CHECKPOINT_PATH"

starcop_dir = os.path.dirname(starcop.__file__)
config_base = os.path.join(starcop_dir, "config.yaml")

config = OmegaConf.load(config_base)

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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ModelModule.load_from_checkpoint(
    checkpoint,
    settings=config,
    map_location=device,
    weights_only=False,
)

model = model.to(device)
model.eval()

x = torch.zeros(
    (1, 4, 512, 512),
    dtype=torch.float32,
    device=device,
)

with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).to(torch.uint8)

print()
print("===== SMOKE TEST =====")
print("Entrada:", tuple(x.shape))
print("Logits :", tuple(logits.shape))
print("Prob   :", tuple(prob.shape))
print("Pred   :", tuple(pred.shape))

assert tuple(x.shape) == (1, 4, 512, 512)
assert tuple(logits.shape) == (1, 1, 512, 512)

print()
print("Smoke test: OK")
PY

ok "Rede executa corretamente."


# ============================================================
# RESUMO FINAL
# ============================================================

echo
echo "============================================================"
echo " AMBIENTE PREPARADO COM SUCESSO"
echo "============================================================"
echo
echo "Estrutura:"
echo
echo "  venv/"
echo "  STARCOP/"
echo "  model/"
echo "  STARCOP_mini/"
echo "  resultados/"
echo "  resultados_dataset/"
echo
echo "Checkpoint:"
echo "  model/final_checkpoint_model.ckpt"
echo
echo "Dataset mini:"
echo "  STARCOP_mini/test_mini10.csv"
echo
echo "Para ativar o ambiente posteriormente:"
echo
echo "  source activate_project.sh"
echo
echo "ou:"
echo
echo "  source venv/bin/activate"
echo "  export PYTHONPATH=\"$STARCOP_SOURCE:\$PYTHONPATH\""
echo
echo "Depois você pode executar:"
echo
echo "  python3 scripts/verificando_rede_roda.py"
echo "  python3 scripts/validando_1imagem_rede.py"
echo "  python3 scripts/validando_1imagem_plotando.py"
echo "  python3 scripts/validando_dataset_completo.py"
echo
echo "============================================================"
