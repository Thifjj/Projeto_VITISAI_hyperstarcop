# HyperSTARCOP na ZCU104

Implantação e benchmark do **HyperSTARCOP `mag1c + RGB`** para segmentação de plumas de metano. O mesmo modelo foi avaliado em três alvos:

- CPU x86 do notebook com PyTorch FP32;
- CPU ARM Cortex-A53 da ZCU104 com ExecuTorch/XNNPACK FP32;
- dois núcleos DPU da ZCU104 com VART e modelo INT8.

O projeto inclui preparação do ambiente, validação de qualidade, quantização, execução embarcada e comparação de FPS e latência.

## Estado do projeto

| Fluxo | Estado | Artefato principal |
|---|---|---|
| Referência PyTorch FP32 | concluído | `model/final_checkpoint_model.ckpt` |
| Rede FP32 independente | concluído | `vitis_ai/float_model/hyperstarcop_network_fp32.pth` |
| DPU INT8 | concluído | `vitis_ai/compiled/zcu104_b4096/hyperstarcop.xmodel` |
| CPU ARM/XNNPACK | concluído | `Arm_zcu104/model/hyperstarcop_xnnpack_fp32.pte` |
| Benchmarks comparativos | concluído | `resultados_zcu104/comparacao_benchmarks_configuracoes.md` |

## Início rápido no notebook

Requisito: Python **3.10.20** disponível como `python3.10`.

```bash
git clone https://github.com/Thifjj/Projeto_VITISAI_hyperstarcop.git
cd Projeto_VITISAI_hyperstarcop
chmod +x setup_environment.sh
./setup_environment.sh
source venv/bin/activate
```

O [`setup_environment.sh`](setup_environment.sh) cria os ambientes, instala as versões fixadas e baixa/valida o `STARCOP_mini`. Não é necessário criar o venv ou instalar o pacote STARCOP manualmente.

Teste funcional:

```bash
python scripts/validando_1imagem_rede.py
```

Benchmark sequencial da CPU:

```bash
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py --profile baseline
```

### Opções do setup

| Comando | Ação |
|---|---|
| `./setup_environment.sh` | prepara `venv/`, `venv_executorch/` e o dataset |
| `./setup_environment.sh --main-only` | prepara somente o ambiente PyTorch principal |
| `./setup_environment.sh --executorch-only` | prepara somente o ambiente de exportação ExecuTorch |
| `./setup_environment.sh --check` | confere os dois ambientes sem alterá-los |

O setup não instala Vitis AI/VART e não modifica a imagem da placa. Essas ferramentas pertencem ao ambiente AMD.

## Organização do repositório

```text
.
├── model/                    # checkpoint e configuração originais
├── scripts/                  # validação e benchmark no notebook
├── vitis_ai/                 # extração, quantização e XModel
├── to_zcu/                   # runner e sweep C++ da DPU
├── Arm_zcu104/               # exportação e runtime CPU ARM
├── resultados_dataset/       # referência de qualidade FP32
├── resultados_cpu/           # medições do notebook
├── resultados_zcu104/        # medições DPU e relatório consolidado
├── setup_environment.sh
├── requirements-environment.txt
└── requirements-executorch.txt
```

Diretórios locais grandes, como `venv/`, `STARCOP_mini/`, sysroot, clone do ExecuTorch e builds cruzados, ficam fora do Git. Os CSVs-resumo e a documentação permanecem versionáveis.

Guias por área:

- [scripts](scripts/README.md): validações e benchmarks no notebook;
- [Vitis AI](vitis_ai/README.md): extração, inspeção e quantização;
- [DPU ZCU104](to_zcu/README.md): compilação, execução e sweep;
- [CPU ARM ZCU104](Arm_zcu104/README.md): ExecuTorch/XNNPACK;
- [resultados](resultados_zcu104/README.md): relatórios e fontes numéricas.

## Modelo e dataset

O modelo é a U-Net com encoder MobileNetV2 do HyperSTARCOP. A entrada tem formato `[1, 4, 512, 512]`:

| Canal | Arquivo |
|---:|---|
| 0 | `mag1c.tif` |
| 1 | `TOA_AVIRIS_640nm.tif` |
| 2 | `TOA_AVIRIS_550nm.tif` |
| 3 | `TOA_AVIRIS_460nm.tif` |

A saída é um mapa de logits `[1, 1, 512, 512]`. A máscara usa `sigmoid(logit) > 0,5`.

O `STARCOP_mini` contém 9 imagens de treino e 9 de teste. Os benchmarks usam as nove entradas de `test_mini10.csv`, repetidas quando é necessário medir mais iterações.

> A quantização INT8 usada neste projeto foi calibrada no notebook com **200 imagens do STARCOP completo**. O `STARCOP_mini` não foi usado na calibração, pois possui somente nove imagens de treino. Para refazer a quantização, obtenha o dataset completo separadamente.

## Fluxo 1 — CPU x86/PyTorch

Ative `venv/` e execute a partir da raiz.

```bash
source venv/bin/activate

# validação das nove imagens
python scripts/validando_rede_minidataset.py

# baseline model-only e end-to-end
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py --profile baseline

# reproduz na CPU as configurações vencedoras da DPU
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py \
  --profile zcu104-comparison

# busca de máxima vazão da CPU
python scripts/sweep_hyperstarcop_cpu.py
```

Saídas principais:

- `resultados_dataset/metricas_globais.csv`;
- `resultados_cpu/hyperstarcop_cpu_zcu104_comparison/`;
- `resultados_cpu/hyperstarcop_cpu_sweep/`.

## Fluxo 2 — extração e Vitis AI

Extração e validação FP32 no ambiente principal:

```bash
source venv/bin/activate
python vitis_ai/scripts/01_comparar_modelo_original_rede_pura.py
python vitis_ai/scripts/02_exportar_rede_pura.py
python vitis_ai/scripts/03_validar_rede_standalone.py
```

Inspeção e quantização devem ser executadas no ambiente Vitis AI compatível:

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

O XModel validado já está em `vitis_ai/compiled/zcu104_b4096/hyperstarcop.xmodel`. Consulte [vitis_ai/README.md](vitis_ai/README.md) antes de reconstruí-lo.

## Fluxo 3 — DPU da ZCU104

Copie `to_zcu/`, o XModel e o `STARCOP_mini` para `/home/root` na placa. Dentro de `to_zcu/`:

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

Busca de desempenho:

```bash
./sweep_hyperstarcop_zcu104 \
  --binary ./hyperstarcop_zcu104_optimized \
  --model /home/root/hyperstarcop.xmodel \
  --dataset /home/root/STARCOP_mini \
  --csv /home/root/STARCOP_mini/test_mini10.csv \
  --out /home/root/hyperstarcop_sweep_results \
  --resume
```

O alvo medido possui 2 × `DPUCZDX8G_ISA1_B4096` a 300 MHz. O fluxo do projeto usa Vitis AI 3.5; o `xdputil` capturado na imagem PetaLinux identifica as bibliotecas VART/XIR instaladas no alvo como 3.0.0.

## Fluxo 4 — CPU ARM da ZCU104

Exporte o PTE no notebook:

```bash
./setup_environment.sh --executorch-only
source venv_executorch/bin/activate
python Arm_zcu104/scripts/transformar_compativel_XNNPACK.py
```

Compile o runner usando o sysroot/runtime já preparados e envie os artefatos:

```bash
./Arm_zcu104/scripts/build_runner_arm.sh

scp -O Arm_zcu104/runtime/hyperstarcop_arm_executorch \
  root@IP_DA_PLACA:/home/root/
scp -O Arm_zcu104/model/hyperstarcop_xnnpack_fp32.pte \
  root@IP_DA_PLACA:/home/root/
```

Na placa, o comando completo está em [Arm_zcu104/README.md](Arm_zcu104/README.md). Esse fluxo usa o runtime C++ AArch64; o pacote Python ExecuTorch não é instalado no PetaLinux.

## Métricas

- `model-only`: mede somente o `forward` do modelo;
- `end-to-end`: inclui leitura dos TIFFs, normalização, filas, inferência, sigmoid e threshold;
- qualidade: precisão, recall, F1, IoU e acurácia globais;
- throughput: `itens concluídos / tempo de parede`;
- latência: tempo individual desde a entrada no pipeline até a conclusão.

Com múltiplos runners, FPS não é o inverso da latência média: várias imagens ficam simultaneamente no pipeline. Use sempre `throughput_fps` para comparar vazão.

## Resultados principais

### Qualidade no STARCOP_mini

| Modelo | Precisão | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 89,2663% | 92,0803% | 90,6515% | 82,9014% |
| ExecuTorch/XNNPACK FP32 | 89,2663% | 92,0803% | 90,6515% | 82,9014% |
| DPU INT8 | 90,9764% | 90,3945% | 90,6845% | 82,9567% |

### Desempenho

| Plataforma | Baseline MO | Baseline E2E | Melhor MO | Melhor E2E |
|---|---:|---:|---:|---:|
| CPU x86 FP32 | 2,072 FPS | 2,010 FPS | 6,076 FPS | 5,305 FPS |
| CPU ARM FP32, 4 threads | 0,497 FPS | 0,464 FPS | 0,497 FPS | 0,464 FPS |
| DPU INT8 | 26,472 FPS | 4,821 FPS | 50,876 FPS | 21,673 FPS |

Os valores ARM são médias de duas execuções para 4 threads. A DPU `model-only` de 50,876 FPS é uma medição de busca com 90 itens; as melhores medições E2E da DPU e da CPU x86 são médias de três execuções com 500 itens.

### Aceleração relativa

Cada valor abaixo é calculado dividindo o FPS da primeira plataforma pelo FPS da segunda. Por exemplo, `DPU / CPU ARM = 102,460×` significa que a DPU entregou aproximadamente 102 vezes mais imagens por segundo.

| Comparação | Baseline MO | Baseline E2E | Melhores MO | Melhores E2E |
|---|---:|---:|---:|---:|
| DPU / CPU ARM com 4 threads | 53,312× | 10,392× | **102,460×** | **46,718×** |
| DPU / CPU x86 | 12,773× | 2,399× | **8,373×** | **4,086×** |
| CPU x86 / CPU ARM com 4 threads | 4,174× | 4,332× | **12,237×** | **11,435×** |

Na comparação de máxima vazão, a DPU chegou a **102,460 vezes o FPS da CPU ARM** em `model-only` e **46,718 vezes** em `end-to-end`. Contra a CPU x86 otimizada, a DPU foi **8,373 vezes** mais rápida em `model-only` e **4,086 vezes** em `end-to-end`.

O ganho obtido ao otimizar cada plataforma foi:

| Plataforma | Melhor MO / baseline MO | Melhor E2E / baseline E2E |
|---|---:|---:|
| CPU x86 | 2,932× | 2,640× |
| CPU ARM, 4 threads / 1 thread | 3,105× | 2,967× |
| DPU | 1,922× | 4,496× |

As comparações de “melhores configurações” medem a capacidade máxima observada e usam graus diferentes de paralelismo. Para uma comparação estritamente sequencial, use as colunas de baseline.

Relatórios:

- [comparação completa de benchmarks e configurações](resultados_zcu104/comparacao_benchmarks_configuracoes.md);
- [relatório detalhado CPU x86 × DPU](resultados_zcu104/relatorio.md);
- [índice dos resultados DPU](resultados_zcu104/README.md).

## Referências

- V. Růžička et al., [Semantic segmentation of methane plumes with hyperspectral machine learning models](https://pmc.ncbi.nlm.nih.gov/articles/PMC10656523/).
- [Código oficial STARCOP](https://github.com/spaceml-org/STARCOP).
- [Modelos STARCOP no Hugging Face](https://huggingface.co/isp-uv-es/starcop).
