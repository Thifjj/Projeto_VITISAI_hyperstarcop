# HyperSTARCOP na CPU ARM da ZCU104

Execução FP32 do HyperSTARCOP nos quatro Cortex-A53 da ZCU104 usando ExecuTorch 1.3.1 e XNNPACK. Esse fluxo é independente da DPU INT8 e funciona na mesma imagem PetaLinux 2022.2 da AMD.

## Estado atual

O fluxo está concluído e validado:

- PTE exportado e comparado com o PyTorch;
- runtime ExecuTorch/XNNPACK compilado para AArch64;
- runner específico do `STARCOP_mini` compilado e executado na placa;
- máscaras PTE idênticas às máscaras FP32 nas nove imagens;
- benchmarks realizados com 1, 2 e 4 threads.

## Uso rápido com os artefatos existentes

No notebook, envie o runner, o modelo e o dataset:

```bash
scp -O Arm_zcu104/runtime/hyperstarcop_arm_executorch \
  root@IP_DA_PLACA:/home/root/
scp -O Arm_zcu104/model/hyperstarcop_xnnpack_fp32.pte \
  root@IP_DA_PLACA:/home/root/
scp -O -r STARCOP_mini root@IP_DA_PLACA:/home/root/
```

Na placa:

```bash
chmod +x /home/root/hyperstarcop_arm_executorch

/home/root/hyperstarcop_arm_executorch \
  --profile all \
  --model /home/root/hyperstarcop_xnnpack_fp32.pte \
  --dataset /home/root/STARCOP_mini \
  --csv /home/root/STARCOP_mini/test_mini10.csv \
  --out /home/root/hyperstarcop_arm_results_threads4 \
  --threads 4 \
  --warmup 3 \
  --model-only-iterations 20 \
  --end-to-end-passes 3
```

Use `--threads 1`, `2` ou `4` e altere o diretório de saída para não misturar execuções.

Para copiar os resultados de volta ao notebook:

```bash
rsync -av root@IP_DA_PLACA:/home/root/hyperstarcop_arm_results_threads4/ \
  Arm_zcu104/reports/hyperstarcop_arm_results_threads4/
```

## Ambiente validado

| Item | Notebook | ZCU104 |
|---|---|---|
| Sistema | Ubuntu 24.04 | PetaLinux 2022.2 `honister` |
| Arquitetura | x86_64 | AArch64, 4 × Cortex-A53 |
| Python | 3.10.20 | 3.9.9, não usado pelo runner |
| PyTorch | 2.12.0 | não instalado |
| ExecuTorch | 1.3.1 | runtime C++ 1.3.1 |
| C/C++ | cross GCC 11 | glibc 2.34, GCC 11.2 |

O Python ExecuTorch fica somente no notebook para exportação. A placa executa binários C++ e não precisa de um novo venv.

## Organização

```text
Arm_zcu104/
├── cmake/toolchain-zcu104.cmake
├── model/hyperstarcop_xnnpack_fp32.pte
├── reports/                         # exportação, qualidade e benchmarks
├── runner/                          # runner C++ do STARCOP_mini
├── runtime/                         # binários AArch64 prontos
├── scripts/
│   ├── build_runner_arm.sh
│   ├── transformar_compativel_XNNPACK.py
│   └── verificar_acuracia_comp_original.py
└── README.md
```

Estes diretórios são locais e ignorados pelo Git:

- `executorch/`: clone oficial da tag `v1.3.1`;
- `sysroot-zcu104/`: headers e bibliotecas copiados da placa;
- `build-executorch-zcu104-v3/`: build cruzado;
- `install-executorch-zcu104-v3/`: instalação CMake usada pelo runner;
- `build-runner-arm/`: build do runner específico.

## Exportar o PTE

Na raiz do projeto:

```bash
./setup_environment.sh --executorch-only
source venv_executorch/bin/activate
python Arm_zcu104/scripts/transformar_compativel_XNNPACK.py
```

O script:

1. reconstrói a U-Net MobileNetV2 de quatro canais;
2. carrega `vitis_ai/float_model/hyperstarcop_network_fp32.pth`;
3. exporta com `torch.export`;
4. aplica o particionador XNNPACK;
5. grava `Arm_zcu104/model/hyperstarcop_xnnpack_fp32.pte`;
6. abre o PTE no runtime local e compara a saída com o PyTorch;
7. atualiza `reports/export_xnnpack.json`.

Não consulte `executorch.__version__`: esse atributo não existe. O setup e o exportador usam `importlib.metadata.version("executorch")`.

Para conferir as nove imagens reais:

```bash
python Arm_zcu104/scripts/verificar_acuracia_comp_original.py
```

## Preparar o sysroot

Esse passo só é necessário para reconstruir o runtime. Como a imagem usada não forneceu um SDK separado, o sysroot foi copiado da própria placa:

```bash
mkdir -p Arm_zcu104/sysroot-zcu104/{lib,usr/lib,usr/include}

rsync -aL root@IP_DA_PLACA:/lib/ \
  Arm_zcu104/sysroot-zcu104/lib/
rsync -aL root@IP_DA_PLACA:/usr/lib/ \
  Arm_zcu104/sysroot-zcu104/usr/lib/
rsync -aL root@IP_DA_PLACA:/usr/include/ \
  Arm_zcu104/sysroot-zcu104/usr/include/
```

O aviso de link sem destino para `/usr/lib/m4/m4-1.4.19` não afeta o runtime.

No Ubuntu, instale o compilador cruzado:

```bash
sudo apt install \
  gcc-11-aarch64-linux-gnu \
  g++-11-aarch64-linux-gnu \
  binutils-aarch64-linux-gnu \
  cmake rsync
```

## Reconstruir o runtime ExecuTorch

Clone exatamente no caminho esperado:

```bash
git clone \
  --branch v1.3.1 \
  --depth 1 \
  --recurse-submodules \
  --shallow-submodules \
  https://github.com/pytorch/executorch.git \
  Arm_zcu104/executorch
```

Configure e compile:

```bash
source venv_executorch/bin/activate

cmake \
  -S Arm_zcu104/executorch \
  -B Arm_zcu104/build-executorch-zcu104-v3 \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/Arm_zcu104/cmake/toolchain-zcu104.cmake" \
  -DCMAKE_INSTALL_PREFIX="$PWD/Arm_zcu104/install-executorch-zcu104-v3" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DPYTHON_EXECUTABLE="$PWD/venv_executorch/bin/python"

cmake --build Arm_zcu104/build-executorch-zcu104-v3 \
  --target install --parallel 8
```

Compile o runner do projeto:

```bash
./Arm_zcu104/scripts/build_runner_arm.sh
```

O script gera `Arm_zcu104/runtime/hyperstarcop_arm_executorch` e mostra arquitetura e SHA-256.

## O que o runner mede

- `model-only`: somente o `forward`, com entrada pronta;
- `end-to-end`: leitura dos quatro TIFFs, normalização, inferência, cópia da saída, sigmoid e threshold;
- qualidade: TP, FP, FN, TN, precisão, recall, F1, IoU e acurácia;
- desempenho: throughput, média, mediana, mínimo, máximo, P90, P95 e P99.

A leitura da label, o cálculo das métricas e a gravação dos CSVs ficam fora da janela temporizada, como no benchmark DPU.

Arquivos gerados:

| Arquivo | Conteúdo |
|---|---|
| `metricas_por_imagem.csv` | qualidade por imagem |
| `metricas_globais.csv` | qualidade após acumular todos os pixels |
| `benchmark_samples.csv` | cada amostra temporal |
| `benchmark_summary.csv` | FPS, latências e tempos dos estágios |

## Resultados medidos

| Threads | `model-only` | Latência MO | `end-to-end` | Latência E2E |
|---:|---:|---:|---:|---:|
| 1 | 0,160 FPS | 6.253,634 ms | 0,156 FPS | 6.394,532 ms |
| 2 | 0,292 FPS | 3.420,646 ms | 0,281 FPS | 3.562,489 ms |
| 4 | **0,497 FPS** | **2.013,936 ms** | **0,464 FPS** | **2.155,551 ms** |

Os valores de 1 e 4 threads são médias de duas execuções; 2 threads foi medido uma vez. A qualidade global foi F1 `0,906515` e IoU `0,829014`, idêntica ao FP32 do notebook.

Consulte [reports/README.md](reports/README.md) para localizar os CSVs e o [relatório comparativo](../resultados_zcu104/comparacao_benchmarks_configuracoes.md) para a análise completa.

## Problemas já resolvidos

| Sintoma | Solução aplicada |
|---|---|
| `executorch` sem `__version__` | usar `importlib.metadata` |
| conflito com PyTorch 2.13 | manter ExecuTorch 1.3.1 com PyTorch 2.12 |
| runtime exigindo glibc mais nova | usar o sysroot PetaLinux e `-nostdinc` |
| `upsample_nearest2d.vec_out` ausente | vincular `portable_ops_lib` como `WHOLE_ARCHIVE` |
| registro duplicado de operadores | não combinar duas bibliotecas que registram o mesmo conjunto |
| `cpuinfo` sem arquivos MIDR | informar `--threads`; as mensagens são informativas |
