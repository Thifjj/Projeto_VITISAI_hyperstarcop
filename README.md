# Validação do HyperSTARCOP com checkpoint oficial do Hugging Face

Este documento descreve, do início ao fim, o procedimento usado para baixar, carregar e validar o modelo **HyperSTARCOP `mag1c + RGB`** do projeto STARCOP, incluindo:

- preparação do ambiente;
- obtenção do código do STARCOP;
- download do checkpoint oficial;
- configuração correta do modelo;
- correção dos problemas encontrados durante o carregamento;
- teste estrutural com uma entrada artificial;
- download do `STARCOP_mini`;
- inferência em uma imagem real;
- cálculo de métricas;
- geração de figuras de visualização;
- validação em todas as imagens disponíveis no `test_mini10.csv`.

---

# 1. Estrutura usada no projeto

A estrutura final ficou aproximadamente assim:

```text
Projeto_Metano/
├── model/
│   ├── final_checkpoint_model.ckpt
│   └── config.yaml
│
├── STARCOP/
│   └── STARCOP/
│       ├── starcop/
│       │   ├── models/
│       │   │   └── model_module.py
│       │   ├── data/
│       │   └── config.yaml
│       └── ...
│
├── STARCOP_mini/
│   ├── test_mini10.csv
│   ├── train_mini10.csv
│   ├── ang2019.../
│   ├── ang2019.../
│   └── ...
│
├── scripts/
│   ├── test_load_checkpoint.py
│   ├── verificando_rede_roda.py
│   ├── validando_1imagem_rede.py
│   ├── validando_1imagem_plotando.py
│   └── validando_dataset_completo.py
│
├── resultados/
├── resultados_dataset/
└── venv/
```

> No caso usado durante os testes, o repositório oficial foi clonado dentro de uma pasta `STARCOP`, resultando em `Projeto_Metano/STARCOP/STARCOP`.

---

# 2. Criar e ativar o ambiente Python

Na raiz do projeto:

```bash
python3 -m venv venv
source venv/bin/activate
```

Atualize o `pip`:

```bash
pip install --upgrade pip
```

Instale as principais dependências:

```bash
pip install \
    torch \
    torchvision \
    pytorch-lightning \
    omegaconf \
    wandb \
    segmentation-models-pytorch \
    torchmetrics \
    fsspec \
    rasterio \
    kornia \
    hydra-core \
    pandas \
    numpy \
    matplotlib \
    gdown \
    huggingface_hub
```

---

# 3. Clonar o STARCOP oficial

Na estrutura usada:

```bash
mkdir -p STARCOP
cd STARCOP

git clone https://github.com/spaceml-org/STARCOP.git
```

Depois volte à raiz:

```bash
cd ..
```

O arquivo importante para o carregamento do modelo deve existir:

```bash
find STARCOP -name "model_module.py"
```

No nosso caso, o resultado foi:

```text
STARCOP/STARCOP/starcop/models/model_module.py
```

---

# 4. Corrigir a importação do pacote STARCOP

Uma instalação simples via `pip` não trouxe corretamente subpacotes como:

```text
starcop/models/model_module.py
starcop/data/normalizer_module.py
```

Por isso usamos diretamente o código-fonte clonado.

Na raiz do projeto:

```bash
export PYTHONPATH="$PWD/STARCOP/STARCOP:$PYTHONPATH"
```

Verifique:

```bash
python3 -c "import starcop; print(starcop.__file__)"
```

O caminho deve apontar para o repositório clonado, aproximadamente:

```text
.../Projeto_Metano/STARCOP/STARCOP/starcop/__init__.py
```

Teste também:

```bash
python3 -c "from starcop.models.model_module import ModelModule; print('ModelModule OK')"
```

Resultado esperado:

```text
ModelModule OK
```

---

# 5. Modelo escolhido

Foi utilizado o modelo:

```text
HyperSTARCOP mag1c + RGB
```

A entrada possui quatro canais:

```text
Canal 0 -> mag1c
Canal 1 -> TOA_AVIRIS_640nm
Canal 2 -> TOA_AVIRIS_550nm
Canal 3 -> TOA_AVIRIS_460nm
```

Arquitetura:

```text
U-Net
└── Encoder: MobileNetV2
```

Saída:

```text
1 canal
```

A saída representa logits de segmentação binária de metano.

---

# 6. Baixar checkpoint e configuração do Hugging Face

Crie a pasta:

```bash
mkdir -p model
```

Use o `huggingface_hub`:

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

repo = "isp-uv-es/starcop"

arquivos = {
    "models/hyperstarcop_mag1c_rgb/config.yaml":
        "model/config.yaml",

    "models/hyperstarcop_mag1c_rgb/final_checkpoint_model.ckpt":
        "model/final_checkpoint_model.ckpt",
}

Path("model").mkdir(exist_ok=True)

for remoto, local in arquivos.items():
    print(f"Baixando {remoto}...")

    src = hf_hub_download(
        repo_id=repo,
        filename=remoto,
        force_download=True
    )

    shutil.copy2(src, local)

    print(f"Salvo em: {local}")

print("Download concluído.")
PY
```

Verifique o checkpoint:

```bash
stat -c "%n -> %s bytes" model/final_checkpoint_model.ckpt
```

O checkpoint real deve ter aproximadamente 80 MB.

---

# 7. Configuração usada para carregar o modelo

O `config.yaml` oficial do checkpoint contém muita metadata do Weights & Biases. Para simplificar a inferência, usamos o `config.yaml` base do STARCOP e sobrescrevemos manualmente apenas os parâmetros necessários.

Exemplo:

```python
import os
import torch
from omegaconf import OmegaConf

import starcop
from starcop.models.model_module import ModelModule

CHECKPOINT = "model/final_checkpoint_model.ckpt"

STARCOP_DIR = os.path.dirname(starcop.__file__)
CONFIG_BASE = os.path.join(STARCOP_DIR, "config.yaml")

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
```

---

# 8. Problema do PyTorch 2.6: `weights_only=True`

Ao tentar carregar o checkpoint antigo apareceu o erro:

```text
_pickle.UnpicklingError: Weights only load failed
```

Isso ocorre porque versões recentes do PyTorch passaram a usar:

```python
weights_only=True
```

como comportamento padrão do `torch.load()`.

O checkpoint original do STARCOP contém objetos adicionais do OmegaConf, por exemplo:

```text
omegaconf.dictconfig.DictConfig
```

Como o checkpoint é proveniente da fonte oficial, usamos:

```python
weights_only=False
```

no carregamento:

```python
model = ModelModule.load_from_checkpoint(
    CHECKPOINT,
    settings=config,
    map_location=device,
    weights_only=False,
)
```

Também foi necessário garantir que nenhuma variável de ambiente estivesse forçando o modo restrito:

```bash
env | grep TORCH_FORCE
```

Caso exista:

```text
TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
```

remova:

```bash
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD
```

---

# 9. Carregar o checkpoint

Código básico:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = ModelModule.load_from_checkpoint(
    CHECKPOINT,
    settings=config,
    map_location=device,
    weights_only=False,
)

model = model.to(device)
model.eval()

print("Checkpoint carregado.")
print("Device:", device)
print("Número de canais:", model.num_channels)
print(model.network)
```

Durante o teste foi exibido:

```text
Lightning automatically upgraded your loaded checkpoint from v1.7.7 to v2.6.5
```

Isso não é erro.

O checkpoint foi carregado com:

```text
Número de canais: 4
```

e a primeira camada mostrou:

```text
Conv2d(4, 32, kernel_size=(3, 3), ...)
```

confirmando que a rede espera quatro canais.

---

# 10. Smoke test da rede

Antes de usar uma imagem real, validamos que toda a rede executava usando uma entrada artificial de zeros:

```python
x = torch.zeros(
    (1, 4, 512, 512),
    dtype=torch.float32,
    device=device
)

with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).to(torch.uint8)

print("Entrada:", x.shape)
print("Logits :", logits.shape)
print("Prob   :", prob.shape)
print("Pred   :", pred.shape)
```

Resultado obtido:

```text
Entrada: torch.Size([1, 4, 512, 512])
Logits : torch.Size([1, 1, 512, 512])
Prob   : torch.Size([1, 1, 512, 512])
Pred   : torch.Size([1, 1, 512, 512])
```

Com uma entrada completamente zerada:

```text
Pixels classificados como metano: 0
```

Esse teste valida apenas o funcionamento estrutural da rede.

---

# 11. Baixar o STARCOP_mini

Como o dataset completo é muito grande, utilizamos o subconjunto `STARCOP_mini`.

Instale o `gdown` caso ainda não esteja instalado:

```bash
pip install gdown
```

Download:

```bash
python3 - <<'PY'
import gdown

gdown.download(
    id="1Qw96Drmk2jzBYSED0YPEUyuc2DnBechl",
    output="STARCOP_mini.zip",
    quiet=False
)
PY
```

Extraia:

```bash
unzip -q STARCOP_mini.zip
```

Conteúdo esperado:

```bash
ls STARCOP_mini
```

Exemplo:

```text
ang20190923t174142_r5826_c168_w151_h151
ang20191018t144405_r2674_c436_w151_h151
...
test_mini10.csv
train_mini10.csv
```

---

# 12. Verificar os arquivos de uma cena

Exemplo:

```bash
ls STARCOP_mini/ang20190923t174142_r5826_c168_w151_h151
```

Foram encontrados, entre outros:

```text
labelbinary.tif
mag1c.tif
TOA_AVIRIS_460nm.tif
TOA_AVIRIS_550nm.tif
TOA_AVIRIS_640nm.tif
weight_mag1c.tif
```

Esses cinco arquivos são suficientes para a validação do HyperSTARCOP `mag1c + RGB`.

---

# 13. Verificar o `test_mini10.csv`

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("STARCOP_mini/test_mini10.csv")

print(df.columns.tolist())
print()
print(df)
PY
```

No conjunto baixado durante o teste, o arquivo chamado `test_mini10.csv` continha 9 linhas.

O código de validação deve usar:

```python
len(df)
```

em vez de assumir manualmente que existem 10 imagens.

---

# 14. Verificar tamanho real das imagens

Apesar dos nomes das pastas terminarem em:

```text
w151_h151
```

os `.tif` presentes no `STARCOP_mini` utilizado tinham tamanho real:

```text
512 x 512
```

Foi confirmado com:

```bash
python3 - <<'PY'
import rasterio

p = "STARCOP_mini/ang20191018t144405_r2674_c436_w151_h151/mag1c.tif"

with rasterio.open(p) as src:
    print("width :", src.width)
    print("height:", src.height)
    print("shape :", src.read(1).shape)
PY
```

Resultado:

```text
width : 512
height: 512
shape : (512, 512)
```

Portanto, nesse conjunto, a imagem pode ser passada diretamente ao modelo.

---

# 15. Inferência em uma imagem real

Leitura dos quatro canais:

```python
from pathlib import Path
import rasterio
import numpy as np

ROOT = Path("STARCOP_mini")

sample_id = "ang20191018t144405_r2674_c436_w151_h151"
folder = ROOT / sample_id

def read_tif(nome):
    path = folder / f"{nome}.tif"

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

mag1c = read_tif("mag1c")
red = read_tif("TOA_AVIRIS_640nm")
green = read_tif("TOA_AVIRIS_550nm")
blue = read_tif("TOA_AVIRIS_460nm")

label = read_tif("labelbinary")
```

Montagem da entrada:

```python
x_np = np.stack(
    [
        mag1c,
        red,
        green,
        blue,
    ],
    axis=0,
)

x = torch.from_numpy(x_np).float()

# [4,512,512] -> [1,4,512,512]
x = x.unsqueeze(0).to(device)
```

Inferência:

```python
with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).to(torch.uint8)
```

Shapes esperados:

```text
Entrada : [1, 4, 512, 512]
Logits  : [1, 1, 512, 512]
Prob    : [1, 1, 512, 512]
Pred    : [1, 1, 512, 512]
```

---

# 16. Normalização da entrada

Não é necessário normalizar manualmente os quatro canais antes de chamar:

```python
model(x)
```

O próprio `ModelModule.forward()` do STARCOP chama internamente:

```python
self.normalizer.normalize_x(x)
```

Portanto, fornecer os dados brutos dos `.tif` ao `ModelModule` mantém o pipeline original.

---

# 17. Cálculo das métricas

Ground truth:

```python
label_np = (label > 0).astype(np.uint8)
pred_np = pred[0, 0].cpu().numpy()
```

Matriz de confusão:

```python
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
```

Precision:

```python
precision = TP / (TP + FP)
```

Recall:

```python
recall = TP / (TP + FN)
```

F1:

```python
f1 = (
    2 * precision * recall /
    (precision + recall)
)
```

IoU:

```python
iou = (
    TP /
    (TP + FP + FN)
)
```

---

# 18. Resultado obtido em uma imagem

A imagem utilizada foi:

```text
ang20191018t144405_r2674_c436_w151_h151
```

Resultado:

```text
Pixels reais: 3171
Pixels previstos: 2737

TP: 2154
FP: 583
FN: 1017
TN: 258390

Precision: 0.7870
Recall:    0.6793
F1:        0.7292
IoU:       0.5738

Prob min: 2.9822084130432884e-20
Prob max: 0.9979161620140076
```

Esse teste confirmou que o checkpoint não apenas carrega, mas também produz uma segmentação coerente em uma imagem real do dataset.

---

# 19. Visualização da inferência

A visualização criada possui seis painéis:

```text
┌────────────────────┬────────────────────┬────────────────────┐
│                    │                    │                    │
│     AVIRIS RGB     │       MAG1C        │    Ground Truth    │
│                    │                    │                    │
├────────────────────┼────────────────────┼────────────────────┤
│                    │                    │                    │
│   Probabilidade    │      Predição      │     TP / FP / FN   │
│                    │                    │                    │
└────────────────────┴────────────────────┴────────────────────┘
```

Mapa de diferenças:

```text
Preto    -> TN
Verde    -> TP
Vermelho -> FP
Azul     -> FN
```

---

# 20. Preparar RGB para visualização

```python
def normalize_image(img, p_low=2, p_high=98):
    low = np.percentile(img, p_low)
    high = np.percentile(img, p_high)

    img = (img - low) / (high - low + 1e-8)

    return np.clip(img, 0, 1)


rgb = np.stack(
    [
        red,
        green,
        blue
    ],
    axis=-1
)

for i in range(3):
    rgb[:, :, i] = normalize_image(
        rgb[:, :, i]
    )
```

MAG1C:

```python
mag1c_vis = normalize_image(
    mag1c,
    p_low=1,
    p_high=99
)
```

---

# 21. Criar mapa TP / FP / FN

```python
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
```

Colormap:

```python
from matplotlib.colors import ListedColormap

difference_cmap = ListedColormap(
    [
        "black",
        "limegreen",
        "red",
        "dodgerblue"
    ]
)
```

---

# 22. Criar a figura

```python
fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

axes[0, 0].imshow(rgb)
axes[0, 0].set_title("AVIRIS RGB")
axes[0, 0].axis("off")

axes[0, 1].imshow(
    mag1c_vis,
    cmap="magma"
)
axes[0, 1].set_title("MAG1C")
axes[0, 1].axis("off")

axes[0, 2].imshow(
    label_np,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[0, 2].set_title("Ground Truth")
axes[0, 2].axis("off")

axes[1, 0].imshow(
    prob_np,
    cmap="inferno",
    vmin=0,
    vmax=1
)
axes[1, 0].set_title("Probabilidade de metano")
axes[1, 0].axis("off")

axes[1, 1].imshow(
    pred_np,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[1, 1].set_title(
    "Predição - threshold 0.5"
)
axes[1, 1].axis("off")

axes[1, 2].imshow(
    differences,
    cmap=difference_cmap,
    vmin=0,
    vmax=3
)
axes[1, 2].set_title(
    "Verde=TP | Vermelho=FP | Azul=FN"
)
axes[1, 2].axis("off")
```

Título:

```python
fig.suptitle(
    f"HyperSTARCOP — {sample_id}\n"
    f"Precision={precision:.4f}   "
    f"Recall={recall:.4f}   "
    f"F1={f1:.4f}   "
    f"IoU={iou:.4f}",
    fontsize=17,
    fontweight="bold"
)
```

Salvar:

```python
output_path = (
    Path("resultados") /
    f"validacao_{sample_id}.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)
```

---

# 23. Problema com `plt.show()`

Ao executar em terminal apareceu:

```text
UserWarning: FigureCanvasAgg is non-interactive,
and thus cannot be shown
```

A imagem havia sido salva corretamente; apenas não podia ser exibida em janela pelo backend atual.

A forma mais simples de visualizar é:

```bash
xdg-open resultados/validacao_<ID>.png
```

No KDE também pode ser usado:

```bash
gwenview resultados/validacao_<ID>.png
```

---

# 24. Validar todas as imagens do `test_mini10.csv`

Em vez de:

```python
row = df.iloc[0]
```

é usado:

```python
for index, row in df.iterrows():
```

Para cada imagem:

1. ler os quatro canais;
2. montar `[1,4,512,512]`;
3. executar `model(x)`;
4. aplicar `sigmoid`;
5. threshold `0.5`;
6. comparar com `labelbinary`;
7. calcular TP, FP, FN, TN;
8. calcular Precision, Recall, F1, IoU;
9. salvar a figura;
10. acumular TP, FP, FN e TN globais.

---

# 25. Acumuladores globais

Antes do loop:

```python
TP_GLOBAL = 0
FP_GLOBAL = 0
FN_GLOBAL = 0
TN_GLOBAL = 0

resultados = []
```

Dentro do loop:

```python
TP_GLOBAL += TP
FP_GLOBAL += FP
FN_GLOBAL += FN
TN_GLOBAL += TN
```

---

# 26. Métricas globais

Após processar todas as imagens:

```python
precision_global = (
    TP_GLOBAL /
    (TP_GLOBAL + FP_GLOBAL)
)

recall_global = (
    TP_GLOBAL /
    (TP_GLOBAL + FN_GLOBAL)
)

f1_global = (
    2 * precision_global * recall_global /
    (precision_global + recall_global)
)

iou_global = (
    TP_GLOBAL /
    (
        TP_GLOBAL +
        FP_GLOBAL +
        FN_GLOBAL
    )
)
```

Essas métricas devem ser consideradas a principal referência para comparar diferentes implementações do modelo.

---

# 27. Métrica global versus média por imagem

Existem duas formas diferentes de resumir os resultados.

## Métrica global

Primeiro soma todos os pixels:

```text
TP_global = TP1 + TP2 + ...
FP_global = FP1 + FP2 + ...
FN_global = FN1 + FN2 + ...
TN_global = TN1 + TN2 + ...
```

Depois calcula:

```text
Precision
Recall
F1
IoU
```

Essa abordagem trata todos os pixels do conjunto como uma única grande amostra.

## Média das imagens

Também pode ser calculado:

```text
F1_medio = (
    F1_imagem1 +
    F1_imagem2 +
    ...
) / N
```

As duas medidas não são equivalentes.

Para comparação futura entre PyTorch, quantizado, Vitis AI e FPGA, a métrica global é uma referência especialmente útil.

---

# 28. Arquivos gerados na validação completa

O script de validação em todas as imagens gera:

```text
resultados_dataset/
├── validacao_<imagem_1>.png
├── validacao_<imagem_2>.png
├── validacao_<imagem_3>.png
├── ...
├── metricas_por_imagem.csv
└── metricas_globais.csv
```

`metricas_por_imagem.csv` contém:

```text
id
pixels_reais
pixels_previstos
TP
FP
FN
TN
precision
recall
f1
iou
accuracy
prob_min
prob_max
```

`metricas_globais.csv` contém:

```text
num_imagens
TP
FP
FN
TN
precision_global
recall_global
f1_global
iou_global
accuracy_global
precision_media
recall_media
f1_media
iou_media
```

---

# 29. Fluxo completo validado

O pipeline final ficou:

```text
Hugging Face
     │
     ├── final_checkpoint_model.ckpt
     └── config.yaml
            │
            ▼
       ModelModule
            │
            ▼
      HyperSTARCOP
            │
   ┌────────┴────────┐
   │                 │
MobileNetV2       U-Net decoder
   │                 │
   └────────┬────────┘
            │
            ▼
      1 canal de logits
            │
          sigmoid
            │
            ▼
       probabilidade
            │
        threshold 0.5
            │
            ▼
       máscara binária
            │
            ├──────────────┐
            │              │
            ▼              ▼
        predição       labelbinary
            │              │
            └──────┬───────┘
                   │
                   ▼
           TP / FP / FN / TN
                   │
                   ▼
      Precision / Recall / F1 / IoU
                   │
                   ▼
         visualização + CSV
```

---

# 30. Estado atual da validação

Até este ponto foi confirmado:

```text
[OK] Código oficial do STARCOP carregado

[OK] ModelModule encontrado

[OK] Checkpoint oficial carregado

[OK] Checkpoint antigo compatível com Lightning atual

[OK] HyperSTARCOP reconstruído

[OK] 4 canais de entrada confirmados

[OK] U-Net + MobileNetV2 confirmados

[OK] Entrada [1,4,512,512] aceita

[OK] Saída [1,1,512,512] obtida

[OK] STARCOP_mini baixado

[OK] Imagem real 512x512 carregada

[OK] Inferência real executada

[OK] Precision calculada

[OK] Recall calculado

[OK] F1 calculado

[OK] IoU calculado

[OK] Ground truth comparado

[OK] Figura de validação gerada

[OK] Pipeline preparado para todas as imagens do CSV
```

---

# 31. Próxima utilização deste pipeline

Esse procedimento agora pode servir como baseline para comparar:

```text
PyTorch FP32 original
        ↓
PyTorch quantizado
        ↓
TensorFlow / Keras
        ↓
Vitis AI INT8
        ↓
XModel
        ↓
DPU na ZCU104
```

O ideal é utilizar exatamente as mesmas imagens e as mesmas métricas em todas as etapas.

Assim, qualquer diferença futura em:

```text
TP
FP
FN
TN
Precision
Recall
F1
IoU
```

poderá ser atribuída à transformação, quantização, compilação ou execução do modelo, e não à mudança do conjunto de validação.

---

# 32. Comandos principais resumidos

Ativar ambiente:

```bash
source venv/bin/activate
```

Adicionar STARCOP ao `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/STARCOP/STARCOP:$PYTHONPATH"
```

Testar import:

```bash
python3 -c "from starcop.models.model_module import ModelModule; print('ModelModule OK')"
```

Testar carregamento:

```bash
python3 scripts/test_load_checkpoint.py
```

Smoke test:

```bash
python3 scripts/verificando_rede_roda.py
```

Validar uma imagem:

```bash
python3 scripts/validando_1imagem_rede.py
```

Gerar visualização:

```bash
python3 scripts/validando_1imagem_plotando.py
```

Validar dataset disponível:

```bash
python3 scripts/validando_dataset_completo.py
```

Abrir uma figura:

```bash
xdg-open resultados/validacao_<ID>.png
```

ou:

```bash
gwenview resultados/validacao_<ID>.png
```

---

# Conclusão

O checkpoint oficial do **HyperSTARCOP `mag1c + RGB`** foi carregado com sucesso e validado em uma imagem real do `STARCOP_mini`.

A rede recebe:

```text
[1, 4, 512, 512]
```

e gera:

```text
[1, 1, 512, 512]
```

O pipeline completo de inferência, comparação com `labelbinary`, cálculo de métricas e geração de visualizações está funcionando e pode ser usado como referência para as próximas etapas de aceleração e comparação com Vitis AI e FPGA.
