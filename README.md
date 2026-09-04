# HyperSTARCOP no Vitis AI e na ZCU104

Implementação, validação e análise de desempenho do **HyperSTARCOP `mag1c + RGB`** para segmentação de plumas de metano. O projeto parte do checkpoint FP32 publicado pelo [STARCOP](https://github.com/spaceml-org/STARCOP), extrai a rede U-Net com encoder MobileNetV2, realiza quantização INT8 e compilação pelo Vitis AI e executa o modelo nos dois núcleos DPU de uma AMD/Xilinx ZCU104.

Além da implantação em FPGA, o repositório contém uma implementação equivalente para CPU. Ela usa as mesmas entradas, normalização, pós-processamento, métricas de qualidade e métodos de medição de desempenho empregados na placa, permitindo uma comparação controlada.

## Início rápido

Com o Python 3.10.20 já instalado, a preparação dos ambientes locais é feita pelo [`setup_environment.sh`](setup_environment.sh):

```bash
git clone https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop.git
cd Projeto_VITISAI_hyperstarcop

chmod +x setup_environment.sh
./setup_environment.sh
source venv/bin/activate
```

Não é necessário criar o `venv`, instalar bibliotecas ou baixar o `STARCOP_mini` manualmente. Depois disso, um primeiro teste pode ser executado com:

```bash
python scripts/validando_1imagem_rede.py
```

Ou o benchmark básico da CPU:

```bash
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py --profile baseline
```

O setup automatiza a preparação local, mas não instala o Vitis AI no sistema nem configura a imagem da ZCU104. Essas etapas dependem do container/runtime oficial da AMD e só são necessárias para reconstruir ou executar a versão acelerada.

Por padrão, o setup mantém dois ambientes separados para evitar conflitos de versões:

| Ambiente | Finalidade |
|---|---|
| `venv/` | checkpoint original, validações e benchmarks PyTorch atuais |
| `venv_executorch/` | exportação e validação do `.pte` com ExecuTorch/XNNPACK |

Para preparar somente um deles:

```bash
./setup_environment.sh --main-only
./setup_environment.sh --executorch-only
```

O ambiente `venv_executorch/` é usado no notebook. A execução na CPU ARM da ZCU104 deverá usar o runtime C++ AArch64, compilado separadamente, e não o Python 3.9.9 da placa.

## Objetivos e trabalho realizado

O fluxo desenvolvido neste projeto cobre:

1. validação do checkpoint oficial do HyperSTARCOP;
2. extração da CNN para um modelo PyTorch independente do Lightning;
3. conferência de equivalência entre o checkpoint e a rede extraída;
4. inspeção da arquitetura para o alvo `DPUCZDX8G_ISA1_B4096`;
5. quantização pós-treinamento de FP32 para INT8;
6. compilação para o formato `.xmodel` da ZCU104;
7. implementação da inferência na placa em C++ com VART e OpenCV;
8. validação da segmentação no `STARCOP_mini`;
9. benchmarks `model-only` e `end-to-end` com `batch = 1`;
10. busca de paralelismo com runners, workers, slots e afinidade;
11. reprodução das medições na CPU do notebook;
12. comparação de qualidade e desempenho entre CPU FP32 e DPU INT8.

## Modelo e dados

O modelo avaliado é o **HyperSTARCOP `mag1c + RGB`**, proposto no artigo [Semantic segmentation of methane plumes with hyperspectral machine learning models](https://pmc.ncbi.nlm.nih.gov/articles/PMC10656523/).

A entrada possui quatro canais de `512 × 512` pixels:

| Canal | Produto |
|---:|---|
| 0 | `mag1c.tif` |
| 1 | `TOA_AVIRIS_640nm.tif` — vermelho |
| 2 | `TOA_AVIRIS_550nm.tif` — verde |
| 3 | `TOA_AVIRIS_460nm.tif` — azul |

A rede produz um mapa de logits com um canal. A máscara binária é obtida por:

```text
logits → sigmoid → threshold 0,5 → máscara de metano
```

O `STARCOP_mini` usado nos testes possui nove imagens de treino e nove de teste. As medições apresentadas neste repositório usam as nove imagens listadas em `test_mini10.csv`.

## Fluxo do projeto

```text
Checkpoint oficial FP32
          │
          ▼
  HyperSTARCOP completo
          │ extração da CNN
          ▼
U-Net + MobileNetV2 FP32 ───────► benchmark na CPU
          │
          ├── inspeção para a DPU
          ├── calibração e quantização INT8
          └── compilação Vitis AI
                      │
                      ▼
             hyperstarcop.xmodel
                      │
                      ▼
          VART C++ na ZCU104 (2 DPUs)
                      │
                      ▼
       métricas, latência e throughput
```

## Organização do repositório

```text
Projeto_Metano/
├── setup_environment.sh          # cria venv, instala libs e baixa o dataset
├── requirements-environment.txt  # dependências do ambiente principal
├── requirements-executorch.txt   # dependências de exportação ExecuTorch
├── model/                        # checkpoint e configuração originais
├── STARCOP_mini/                 # dataset reduzido, criado pelo setup
├── scripts/                      # validação e benchmarks em CPU
├── vitis_ai/
│   ├── scripts/                  # extração, inspeção e quantização
│   ├── float_model/              # pesos FP32 da CNN independente
│   ├── inspect/                  # relatório de compatibilidade da DPU
│   ├── quantized_int8_calib200/  # artefatos quantizados
│   ├── compiled/zcu104_b4096/    # XModel compilado
│   └── results_*/                # validações intermediárias
├── to_zcu/                       # aplicação C++ e sweep para a placa
├── resultados_dataset/           # métricas e imagens da referência FP32
├── resultados_cpu/               # benchmarks realizados no notebook
└── resultados_zcu104/            # resultados, logs e relatório consolidado
```

Os diretórios `venv/`, `STARCOP_mini/` e os arquivos grandes podem não aparecer em um clone limpo, pois são preparados localmente ou distribuídos como artefatos.

## Uso básico no notebook/CPU

### 1. Pré-requisito

É necessário ter o **Python 3.10.20** instalado e acessível como `python3.10`. O script não altera o Python do sistema.

### 2. Preparar automaticamente o ambiente

Na raiz do repositório:

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

Sem opções, esse único comando prepara `venv/` e `venv_executorch/`. Para o ambiente principal, ele:

- cria o `venv` usando exatamente Python 3.10.20;
- atualiza `pip`, `setuptools` e `wheel` dentro do ambiente;
- instala as versões fixadas em `requirements-environment.txt`;
- baixa e extrai o `STARCOP_mini`, se necessário;
- valida os CSVs e os arquivos usados pelo modelo;
- verifica se existem dependências Python quebradas.

Em resumo:

| Etapa | Feita pelo setup? |
|---|---|
| Criar ou atualizar o `venv` | Sim |
| Instalar as bibliotecas nas versões corretas | Sim |
| Instalar o pacote Python `starcop` | Sim |
| Baixar, extrair e validar o `STARCOP_mini` | Sim |
| Criar ambiente separado para ExecuTorch/XNNPACK | Sim |
| Gerar o arquivo `.pte` | Não: será feito pelo script de exportação |
| Baixar o checkpoint FP32 | Não é necessário: já está em `model/` |
| Executar validações e benchmarks | Não: o usuário escolhe qual teste executar |
| Instalar Vitis AI/VART | Não: depende do ambiente oficial da AMD |
| Configurar a imagem e o hardware da ZCU104 | Não |

Para apenas conferir uma instalação existente, sem modificar arquivos:

```bash
./setup_environment.sh --check
./setup_environment.sh --main-only --check
./setup_environment.sh --executorch-only --check
```

Ative o ambiente depois da preparação:

```bash
source venv/bin/activate
```

### 3. Testes rápidos

Carregar o checkpoint original:

```bash
python scripts/test_load_checkpoint.py
```

Executar um teste estrutural com entrada artificial:

```bash
python scripts/verificando_rede_roda.py
```

Validar uma imagem real e mostrar suas métricas:

```bash
python scripts/validando_1imagem_rede.py
```

Validar o subconjunto de teste e gerar visualizações:

```bash
python scripts/validando_rede_minidataset.py
```

### 4. Benchmark de CPU

Executar o baseline sequencial FP32:

```bash
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py \
  --profile baseline
```

Reproduzir na CPU as configurações selecionadas na campanha da ZCU104:

```bash
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py \
  --profile zcu104-comparison
```

Buscar automaticamente a maior vazão da CPU e confirmar as melhores configurações com três repetições:

```bash
python scripts/sweep_hyperstarcop_cpu.py
```

Essa busca pode demorar, pois executa dezenas de combinações. Os resultados são gravados em `resultados_cpu/hyperstarcop_cpu_sweep/` e podem ser retomados sem repetir execuções válidas.

## Fluxo Vitis AI

Os passos abaixo são necessários somente para reconstruir o modelo INT8/XModel. Eles devem ser executados no ambiente Vitis AI compatível; o `setup_environment.sh` prepara o ambiente Python local, mas não instala o toolchain proprietário da AMD.

### 1. Extrair e validar a CNN

```bash
python vitis_ai/scripts/01_comparar_modelo_original_rede_pura.py
python vitis_ai/scripts/02_exportar_rede_pura.py
python vitis_ai/scripts/03_validar_rede_standalone.py
```

O resultado principal é:

```text
vitis_ai/float_model/hyperstarcop_network_fp32.pth
```

### 2. Inspecionar e quantizar

No container Vitis AI:

> **Dataset de calibração:** a quantização INT8 deste projeto foi realizada localmente, no notebook, com **200 imagens do dataset STARCOP completo**. O `STARCOP_mini` não foi usado para calibrar o quantizador, pois contém somente 9 imagens de treino e 9 de teste. Ele foi usado posteriormente para validar a qualidade do modelo quantizado e executar os benchmarks reproduzíveis na CPU e na ZCU104. O `setup_environment.sh` baixa apenas o `STARCOP_mini`; para refazer a calibração é necessário obter separadamente o dataset completo e informar seu caminho em `--dataset_root`.

```bash
python vitis_ai/scripts/04_inspecionar_vitis_ai.py

python vitis_ai/scripts/05_quantizar_int8.py \
  --quant_mode calib \
  --calib_samples 200 \
  --dataset_root /workspace/dataset_STARCOP

python vitis_ai/scripts/05_quantizar_int8.py \
  --quant_mode test \
  --calib_samples 200 \
  --dataset_root /workspace/dataset_STARCOP \
  --deploy
```

A opção `--calib_samples 200` seleciona 200 amostras do `train.csv` presente no dataset completo indicado por `--dataset_root`. O XModel compilado utilizado na placa está em:

```text
vitis_ai/compiled/zcu104_b4096/hyperstarcop.xmodel
```

## Execução na ZCU104

A placa usada possui dois núcleos `DPUCZDX8G_ISA1_B4096` a 300 MHz e runtime Vitis AI/VART 3.5.0.

Copie para a ZCU104:

- `to_zcu/hyperstarcop_zcu104_optimized.cpp`;
- `to_zcu/sweep_hyperstarcop_zcu104.cpp`;
- `to_zcu/build_hyperstarcop_zcu104_optimized.sh`;
- `vitis_ai/compiled/zcu104_b4096/hyperstarcop.xmodel`;
- `STARCOP_mini/`.

Na placa, coloque os três arquivos de `to_zcu/` no mesmo diretório e execute:

```bash
chmod +x build_hyperstarcop_zcu104_optimized.sh
./build_hyperstarcop_zcu104_optimized.sh
```

Benchmark básico:

```bash
./hyperstarcop_zcu104_optimized \
  --profile all \
  --model /home/root/hyperstarcop.xmodel \
  --dataset /home/root/STARCOP_mini \
  --csv /home/root/STARCOP_mini/test_mini10.csv
```

Busca automática das melhores configurações:

```bash
./sweep_hyperstarcop_zcu104 \
  --binary ./hyperstarcop_zcu104_optimized \
  --model /home/root/hyperstarcop.xmodel \
  --dataset /home/root/STARCOP_mini \
  --csv /home/root/STARCOP_mini/test_mini10.csv \
  --out /home/root/hyperstarcop_sweep_results \
  --resume
```

Detalhes dos parâmetros e das etapas do sweep estão em [`to_zcu/README_SWEEP.md`](to_zcu/README_SWEEP.md).

## Métricas e metodologia

### Qualidade da segmentação

As máscaras são comparadas pixel a pixel com `labelbinary.tif`. São acumulados TP, FP, FN e TN e, a partir deles, calculados precisão, recall, F1, IoU e acurácia. A comparação principal usa as métricas globais: primeiro somam-se os pixels das nove imagens e depois calculam-se os indicadores.

### Desempenho

São reportados dois modos:

- **`model-only`**: cronometra somente a inferência; a entrada já está carregada e normalizada;
- **`end-to-end`**: inclui leitura dos quatro TIFFs, normalização, filas, inferência, sigmoid e limiarização.

O throughput é calculado pela vazão total do sistema:

```text
throughput_fps = imagens concluídas / tempo total de parede
```

A latência é medida individualmente da chegada da imagem ao término do pipeline. Em configurações concorrentes, diversas imagens são processadas simultaneamente; portanto, a latência média não é simplesmente o inverso do throughput. Todas as medições usam `batch = 1`.

## Resultados obtidos

### Qualidade: FP32, INT8 e artigo

| Referência | Conjunto/recorte | Precisão | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|
| CPU FP32 deste projeto | 9 imagens do `STARCOP_mini` | 89,2663% | 92,0803% | 90,6515% | 82,9014% |
| ZCU104 INT8 deste projeto | mesmas 9 imagens | 90,9764% | 90,3945% | 90,6845% | 82,9567% |
| Artigo — HyperSTARCOP `mag1c + RGB` | teste completo, plumas fortes | — | — | 81,96 ± 3,71% | — |
| Artigo — HyperSTARCOP `mag1c + RGB` | teste completo, plumas fracas | — | — | 43,42 ± 5,72% | — |

A quantização INT8 preservou a qualidade no conjunto avaliado: em relação ao FP32, o F1 variou apenas **+0,033 ponto percentual** e o IoU **+0,055 ponto percentual**.

Os valores do artigo não são uma comparação direta com os 90,68% medidos aqui. O artigo usa o conjunto completo, separa eventos fortes e fracos e apresenta a média de cinco treinamentos; este projeto usa o checkpoint oficial fixo e as nove imagens com pluma do `STARCOP_mini`. A tabela serve para contextualizar o resultado, sem afirmar superioridade sobre o artigo.

### Desempenho: CPU versus ZCU104

| Cenário | CPU FP32 | ZCU104 INT8 | Aceleração da ZCU104 |
|---|---:|---:|---:|
| Baseline `model-only` | 2,072 FPS | 26,472 FPS | 12,773× |
| Baseline `end-to-end` | 2,010 FPS | 4,821 FPS | 2,399× |
| Melhor configuração própria `model-only` | 6,076 FPS | 50,876 FPS | 8,373× |
| Melhor configuração própria `end-to-end` | 5,305 FPS | 21,673 FPS | 4,086× |

As melhores configurações foram:

- CPU `model-only`: 4 runners × 2 threads, usando 8 núcleos físicos;
- CPU `end-to-end`: 8 runners × 1 thread, 2 workers de pré-processamento e 8 slots;
- ZCU104 `model-only`: 4 runners e 1 slot por runner;
- ZCU104 `end-to-end`: 3 runners, 4 workers de pré-processamento, 16 de pós-processamento e 5 slots por runner.

Na comparação restrita a dois núcleos físicos do notebook, um runner com duas threads alcançou **3,770 FPS** em `model-only` e **3,540 FPS** em `end-to-end`. Isso não torna núcleos CPU e DPU arquiteturalmente equivalentes, mas oferece uma referência adicional para o paralelismo disponível.

Os dados completos, latências, percentis, resultados por imagem e explicação detalhada do cálculo de FPS estão em [`resultados_zcu104/relatorio.md`](resultados_zcu104/relatorio.md).

## Referências

- V. Růžička et al., [Semantic segmentation of methane plumes with hyperspectral machine learning models](https://pmc.ncbi.nlm.nih.gov/articles/PMC10656523/).
- [Código oficial STARCOP](https://github.com/spaceml-org/STARCOP).
- [Checkpoint e modelos STARCOP no Hugging Face](https://huggingface.co/isp-uv-es/starcop).
- [Repositório deste projeto no GitHub](https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop).
