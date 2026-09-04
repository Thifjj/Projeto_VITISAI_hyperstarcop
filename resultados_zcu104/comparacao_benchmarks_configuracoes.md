# Comparação completa dos benchmarks do HyperSTARCOP

Data da consolidação: 4 de setembro de 2026.

## 1. Objetivo e escopo

Este relatório reúne os benchmarks encontrados no repositório para três formas de execução do HyperSTARCOP:

- CPU x86 do notebook com PyTorch e modelo FP32;
- CPU ARM Cortex-A53 da ZCU104 com ExecuTorch, XNNPACK e modelo FP32;
- DPU da ZCU104 com VART e modelo quantizado em INT8.

Também são incluídas as campanhas de busca de configuração da CPU x86 e da DPU. Todas as medições usam `batch = 1`, entrada com quatro canais em `512 × 512`, saída binária em `512 × 512` e o `STARCOP_mini` com nove imagens de teste. Os resultados de desempenho repetem essas imagens até completar o número de amostras de cada ensaio.

Os resultados de qualidade usam `sigmoid > 0,5`. A quantização INT8 foi calibrada separadamente com 200 imagens do STARCOP completo; o conjunto mini foi usado para validação e benchmark, não para calibração.

Nas tabelas, `R` significa número de runners, `T` é o número de threads internas por runner, `Pré` e `Pós` são os workers de pré e pós-processamento, `SPR` é a quantidade de slots por runner, `MO` significa `model-only` e `E2E` significa `end-to-end`. `Pin` indica afinidade explícita de CPU.

## 2. Resumo dos principais resultados

### 2.1 Baseline sequencial

O baseline é o cenário com uma imagem em execução por vez. Na CPU x86 e na DPU ele usa um runner, um worker de pré-processamento, um de pós-processamento e um slot. Na CPU ARM, o runner é único e varia apenas o número de threads internas do XNNPACK.

| Plataforma | Precisão | Configuração | `model-only` FPS | Latência média | `end-to-end` FPS | Latência média |
|---|---|---|---:|---:|---:|---:|
| CPU x86 do notebook | FP32 | R1, T1, Pré1, Pós1, SPR1, sem afinidade | 2,072 | 482,515 ms | 2,010 | 497,361 ms |
| CPU ARM ZCU104 | FP32 | R1, XNNPACK 1 thread | 0,160 | 6.253,634 ms | 0,156 | 6.394,532 ms |
| CPU ARM ZCU104 | FP32 | R1, XNNPACK 2 threads | 0,292 | 3.420,646 ms | 0,281 | 3.562,489 ms |
| CPU ARM ZCU104 | FP32 | R1, XNNPACK 4 threads | **0,497** | **2.013,936 ms** | **0,464** | **2.155,551 ms** |
| DPU ZCU104 | INT8 | R1, Pré1, Pós1, SPR1, sem afinidade | **26,472** | **37,776 ms** | **4,821** | **207,426 ms** |

Para ARM, os valores de 1 e 4 threads são a média de duas execuções; 2 threads possui uma execução. Os resultados brutos estão detalhados na seção 7.

### 2.2 Maior vazão encontrada em cada plataforma

| Plataforma | Modo | Configuração vencedora | Validação final | FPS | Latência média | P95 | P99 |
|---|---|---|---|---:|---:|---:|---:|
| CPU x86 | `model-only` | R4 × T2, 4 slots, 8 núcleos físicos, sem pin | média de 3 × 500 | 6,076 | 657,373 ms | 744,972 ms | 800,588 ms |
| CPU x86 | `end-to-end` | R8 × T1, Pré2, Pós1, SPR1, 8 slots, 16 CPUs lógicas, com pin | média de 3 × 500 | 5,305 | 1.837,972 ms | 2.387,604 ms | 2.585,550 ms |
| CPU ARM | `model-only` | R1, XNNPACK 4 threads | média de 2 × 20 | 0,497 | 2.013,936 ms | 2.016,813 ms | 2.021,906 ms |
| CPU ARM | `end-to-end` | R1, XNNPACK 4 threads | média de 2 × 27 | 0,464 | 2.155,551 ms | 2.178,781 ms | 2.180,217 ms |
| DPU ZCU104 | `model-only` | R4, SPR1, 4 slots, sem pin | 1 × 90 | 50,876 | 77,670 ms | 82,313 ms | 82,840 ms |
| DPU ZCU104 | `end-to-end` | R3, Pré4, Pós16, SPR5, 15 slots, com pin | média de 3 × 500 | 21,673 | 262,386 ms | 315,122 ms | 339,343 ms |

O melhor `model-only` da DPU é uma medição de busca com 90 itens e não foi repetido três vezes. Já os melhores resultados finais `end-to-end` da DPU e os dois modos da CPU x86 são médias de três execuções de 500 itens.

Comparando as melhores configurações específicas de cada plataforma:

| Comparação | `model-only` | `end-to-end` |
|---|---:|---:|
| DPU contra CPU x86 | 8,373× mais FPS | 4,086× mais FPS |
| DPU contra CPU ARM com 4 threads | 102,460× mais FPS | 46,718× mais FPS |
| Ganho da busca na CPU x86 sobre seu baseline | 2,932× | 2,640× |
| Ganho da busca na DPU sobre seu baseline | 1,922× | 4,496× |

Essas razões de máxima vazão comparam configurações diferentes e mostram a capacidade máxima observada, não uma equivalência núcleo a núcleo.

## 3. Ambientes e modelos utilizados

| Item | CPU x86 do notebook | CPU ARM da ZCU104 | DPU da ZCU104 |
|---|---|---|---|
| Processador/acelerador | AMD Ryzen 9 5980HX, 8 núcleos e 16 CPUs lógicas | Quad-core ARM Cortex-A53 AArch64 | 2 × DPUCZDX8G ISA1 B4096 |
| Sistema | Linux 7.0.0-30-generic, glibc 2.39 | PetaLinux 2022.2, glibc 2.34 | PetaLinux 2022.2, kernel 5.15.36 |
| Runtime | Python 3.10.20, PyTorch 2.13.0+cu130 | ExecuTorch 1.3.1 + XNNPACK, runner C++ AArch64 | VART/XIR; bibliotecas do alvo identificadas pelo `xdputil` como 3.0.0 |
| Modelo | checkpoint `.pth` original | `.pte` delegado ao XNNPACK | `.xmodel` compilado para DPU |
| Precisão | FP32 | FP32 | INT8 |
| Forma da entrada | `[1, 4, 512, 512]` | `[1, 4, 512, 512]` | `[1, 512, 512, 4]` NHWC |
| Forma da saída | `[1, 1, 512, 512]` | `[1, 1, 512, 512]` | `[1, 512, 512, 1]` NHWC |
| Parâmetros | 6.629.233 | mesmo modelo lógico | mesmo modelo lógico quantizado |
| Paralelismo testado | runners, threads PyTorch, workers, slots e afinidade | 1 runner; 1, 2 e 4 threads XNNPACK | runners VART, workers, slots e afinidade |

O código do benchmark DPU se identifica como Vitis AI 3.5, enquanto o inventário capturado no alvo informa bibliotecas VART/XIR 3.0.0. Por isso, este relatório registra separadamente a versão do fluxo do projeto e a versão efetivamente mostrada pelo `xdputil`, sem assumir que sejam iguais.

O PTE possui 26.500.100 bytes. Na verificação da exportação, o erro absoluto máximo contra o modelo PyTorch foi `4,3869 × 10⁻⁵`, e o teste `allclose(rtol=1e-4, atol=1e-5)` passou. Nas nove imagens reais, as máscaras binárias FP32 e PTE foram idênticas pixel a pixel.

## 4. Qualidade da segmentação

| Plataforma/modelo | Imagens | TP | FP | FN | TN | Precisão | Recall | F1 | IoU | Acurácia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU x86 PyTorch FP32 | 9 | 40.310 | 4.847 | 3.467 | 2.310.672 | 0,892663 | 0,920803 | 0,906515 | 0,829014 | 0,996476 |
| CPU ARM ExecuTorch FP32 | 9 | 40.310 | 4.847 | 3.467 | 2.310.672 | 0,892663 | 0,920803 | 0,906515 | 0,829014 | 0,996476 |
| DPU VART INT8 | 9 | 39.572 | 3.925 | 4.205 | 2.311.594 | 0,909764 | 0,903945 | 0,906845 | 0,829567 | 0,996554 |

O ExecuTorch preservou exatamente a máscara do FP32 nessas imagens. Em relação ao FP32, o INT8 aumentou a precisão em 0,017101, reduziu o recall em 0,016858 e manteve F1 e IoU praticamente inalterados: deltas de +0,000330 e +0,000552, respectivamente.

## 5. Benchmarks da CPU x86

### 5.1 Baseline e reprodução das configurações da DPU

Esta campanha aplicou na CPU as mesmas quantidades de runners, workers e slots usadas na ZCU104. Cada runner da CPU usa uma thread intra-op e uma inter-op.

| Cenário | Modo | Configuração | Itens | FPS | Latência média | P95 | P99 |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline | `model-only` | R1, T1, 1 slot, sem pin | 100 | 2,072 | 482,515 ms | 497,163 ms | 514,587 ms |
| Baseline | `end-to-end` | R1, T1, Pré1, Pós1, SPR1, sem pin | 45 | 2,010 | 497,361 ms | 520,913 ms | 523,861 ms |
| Configuração do melhor DPU MO | `model-only` | R4, T1, 4 slots, sem pin | 90 | 5,278 | 749,623 ms | 838,992 ms | 885,486 ms |
| Configuração do melhor DPU E2E | `end-to-end` | R3, T1, Pré4, Pós16, SPR5, 15 slots, com pin | 500 | 4,050 | 4.587,699 ms | 5.234,814 ms | 5.357,743 ms |

### 5.2 Comparação limitada a dois núcleos físicos

As CPUs lógicas `[0, 2]` foram escolhidas para representar dois núcleos físicos distintos do Ryzen.

| Modo | Configuração | Itens | FPS | Latência média | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|
| `model-only` | R1 × T2, 1 slot, sem pin | 90 | **3,770** | **265,233 ms** | **296,243 ms** | **363,273 ms** |
| `model-only` | R2 × T1, 2 slots, com pin | 90 | 3,445 | 576,997 ms | 629,236 ms | 665,202 ms |
| `end-to-end` | R1 × T2, Pré1, Pós1, SPR1, sem pin | 90 | **3,540** | **549,096 ms** | **590,396 ms** | **593,283 ms** |
| `end-to-end` | R2 × T1, Pré1, Pós1, SPR1, com pin | 90 | 3,477 | 833,756 ms | 1.022,007 ms | 1.071,864 ms |

Com a mesma contagem de duas threads/núcleos de propósito geral, o notebook entregou 12,896× o FPS da CPU ARM em `model-only` e 12,613× em `end-to-end`. Isso compara as CPUs, mas não implica equivalência entre as microarquiteturas.

### 5.3 Campanha completa de busca da CPU

A busca contém 37 execuções válidas. Os conjuntos abaixo descrevem os valores realmente explorados em cada fase; eles não devem ser interpretados como produto cartesiano de todas as opções. A linha exata de cada execução está em [`all_runs.csv`](../resultados_cpu/hyperstarcop_cpu_sweep/all_runs.csv).

| Fase | Execuções | Configurações avaliadas | Melhor resultado da fase |
|---|---:|---|---|
| `10_two_core` | 4 | R1×T2 e R2×T1; 2 núcleos; MO/E2E; SPR1 | MO R1×T2: 3,770 FPS |
| `20_model_search` | 13 | R={1,2,4,8,16}; T={1,2,4,8,16}; 1 a 16 CPUs; SPR1; com/sem pin | R4×T2, 8 CPUs, sem pin: 5,796 FPS |
| `30_e2e_runners` | 5 | R={1,2,4,8,16}; T1; Pré2; Pós1; SPR2; 16 CPUs; pin | R8: 5,161 FPS |
| `40_e2e_refine` | 9 | R8×T1; Pré={1,2,4,8}; Pós={1,2,4}; SPR={1,2,4}; com/sem pin | Pré2, Pós1, SPR1, pin: 5,193 FPS |
| `50_final` | 6 | duas vencedoras; 3 repetições de 500 por modo | MO 6,076 FPS; E2E 5,305 FPS em média |

Resultados finais da busca:

| Modo | Configuração | Repetições | FPS médio | Mediana | Faixa | Latência média | P95 médio | P99 médio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `model-only` | R4×T2, 4 slots, 8 núcleos, sem pin | 3 × 500 | 6,076 | 6,079 | 6,060–6,090 | 657,373 ms | 744,972 ms | 800,588 ms |
| `end-to-end` | R8×T1, Pré2, Pós1, SPR1, 8 slots, 16 CPUs, pin | 3 × 500 | 5,305 | 5,278 | 5,271–5,365 | 1.837,972 ms | 2.387,604 ms | 2.585,550 ms |

### 5.4 Snapshot anterior `hyperstarcop_cpu_optimized`

Este conjunto é mantido como resultado histórico e não substitui a campanha final. Foi executado com PyTorch usando uma thread por runner, aquecimento global de 10, aquecimento de 5 por worker e afinidade habilitada.

| Modo | Configuração | Itens | FPS | Latência média | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|
| Baseline MO | R1×T1, 1 slot | 100 | 2,001 | 499,617 ms | 509,453 ms | 515,567 ms |
| Baseline E2E | R1×T1, Pré1, Pós1, 1 slot | 45 | 1,948 | 512,990 ms | 520,748 ms | 526,037 ms |
| Máximo MO daquele ensaio | R4×T1, 4 slots | 500 | 4,566 | 873,738 ms | 998,912 ms | 1.042,202 ms |
| Máximo E2E daquele ensaio | R4×T1, Pré2, Pós1, 8 slots | 500 | 4,911 | 2.815,645 ms | 3.186,978 ms | 3.261,942 ms |

As diferenças entre esse snapshot e a campanha posterior mostram por que o resultado oficial deve vir das repetições finais, e não apenas de uma execução isolada.

## 6. Benchmarks da DPU ZCU104

### 6.1 Benchmark de validação e baseline

| Modo | Configuração | Itens | FPS | Latência média | P95 | P99 |
|---|---|---:|---:|---:|---:|---:|
| `model-only` | R1, 1 slot | 100 | 26,472 | 37,776 ms | 37,833 ms | 37,848 ms |
| `end-to-end` | R1, Pré1, Pós1, SPR1 | 45 | 4,821 | 207,426 ms | 232,511 ms | 232,591 ms |

A campanha de busca repetiu o baseline E2E com 90 itens e obteve 4,861 FPS, latência média de 205,704 ms, P95 de 230,674 ms e P99 de 230,799 ms. A proximidade confirma a estabilidade do baseline.

### 6.2 Campanha completa de busca da DPU

A busca contém 47 execuções válidas. Os valores exatos estão em [`all_runs.csv`](hyperstarcop_sweep_results/all_runs.csv), e a classificação das execuções finais está em [`ranking_final_runs.csv`](hyperstarcop_sweep_results/ranking_final_runs.csv).

| Fase | Execuções | Configurações avaliadas | Melhor resultado da fase |
|---|---:|---|---|
| `00_baseline` | 1 | R1, Pré1, Pós1, SPR1, sem pin, 90 itens | 4,861 FPS E2E |
| `10_model_runners` | 8 | R={1,2,3,4}; SPR1; com/sem pin; 90 itens | R4 sem pin: 50,876 FPS MO |
| `20_e2e_runners` | 4 | R={3,4}; Pré2; Pós1; SPR2; com/sem pin | R3 sem pin: 12,898 FPS |
| `30_pre_workers` | 5 | R3; Pré={1,2,4,8,16}; Pós1; SPR2; sem pin | Pré4: 19,304 FPS |
| `40_post_workers` | 5 | R3; Pré4; Pós={1,2,4,8,16}; SPR2; sem pin | Pós16: 20,492 FPS |
| `50_slots` | 3 | R3; Pré4; Pós16; SPR={1,2,5}; sem pin | SPR2: 20,547 FPS |
| `55_refine` | 12 | R={3,4}; Pré={4,16}; Pós={8,16}; SPR={2,5}; sem pin | R3/Pré4/Pós16/SPR5: 20,919 FPS |
| `60_final_c1` | 3 | R3, Pré4, Pós16, SPR5, sem pin; 500 itens | média 21,197 FPS |
| `60_final_c2` | 3 | R3, Pré4, Pós16, SPR5, com pin; 500 itens | média 21,673 FPS |
| `60_final_c3` | 3 | R4, Pré4, Pós8, SPR2, sem pin; 500 itens | média 21,109 FPS |

Comparação das três candidatas finais E2E:

| Candidata | Configuração | FPS médio | Mediana | Faixa | Latência média | P95 médio | P99 médio |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | R3, Pré4, Pós16, SPR5, 15 slots, sem pin | 21,197 | 21,134 | 21,003–21,454 | 266,600 ms | 325,451 ms | 363,875 ms |
| C2 — selecionada | R3, Pré4, Pós16, SPR5, 15 slots, com pin | **21,673** | **21,676** | 21,636–21,709 | **262,386 ms** | **315,122 ms** | **339,343 ms** |
| C3 | R4, Pré4, Pós8, SPR2, 8 slots, sem pin | 21,109 | 21,168 | 20,989–21,171 | 264,274 ms | 325,425 ms | 365,264 ms |

A regra registrada em `best_config.json` seleciona a maior mediana de throughput, usando P99 como desempate. A reprodução adicional da C2, com 500 itens, obteve 21,687 FPS, latência média de 261,924 ms, P95 de 311,402 ms e P99 de 343,167 ms.

## 7. Benchmarks da CPU ARM com ExecuTorch/XNNPACK

O runner ARM é sequencial: existe uma chamada `forward` ativa e o XNNPACK distribui o trabalho interno entre 1, 2 ou 4 threads. Foram usados três aquecimentos, 20 inferências `model-only` e três passagens sobre as nove imagens, totalizando 27 itens `end-to-end` por execução.

| Threads | Modo | Execuções | Itens totais | FPS médio | Faixa de FPS | Latência média | P95 médio | P99 médio |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `model-only` | 2 | 40 | 0,160 | 0,1597–0,1601 | 6.253,634 ms | 6.257,416 ms | 6.257,537 ms |
| 1 | `end-to-end` | 2 | 54 | 0,156 | 0,1563–0,1565 | 6.394,532 ms | 6.419,399 ms | 6.419,956 ms |
| 2 | `model-only` | 1 | 20 | 0,292 | 0,2923 | 3.420,646 ms | 3.422,957 ms | 3.423,036 ms |
| 2 | `end-to-end` | 1 | 27 | 0,281 | 0,2807 | 3.562,489 ms | 3.587,070 ms | 3.589,124 ms |
| 4 | `model-only` | 2 | 40 | 0,497 | 0,4945–0,4986 | 2.013,936 ms | 2.016,813 ms | 2.021,906 ms |
| 4 | `end-to-end` | 2 | 54 | 0,464 | 0,4622–0,4657 | 2.155,551 ms | 2.178,781 ms | 2.180,217 ms |

Escalabilidade observada em relação a uma thread:

| Threads ARM | Ganho `model-only` | Ganho `end-to-end` | Eficiência MO | Eficiência E2E |
|---:|---:|---:|---:|---:|
| 2 | 1,828× | 1,795× | 91,4% | 89,7% |
| 4 | 3,105× | 2,967× | 77,6% | 74,2% |

O uso de quatro threads foi o melhor entre as configurações testadas. Não houve busca com múltiplos runners ARM, portanto este relatório não extrapola um resultado de máxima concorrência para essa plataforma.

Os dois registros de 1 thread estão em [`threads1`](../Arm_zcu104/reports/hyperstarcop_arm_results_threads1/benchmark_summary.csv) e no CSV de [`threads4`](../Arm_zcu104/reports/hyperstarcop_arm_results_threads4/benchmark_summary.csv); o segundo arquivo também contém as duas repetições de 4 threads. O resultado de 2 threads está em [`threads2`](../Arm_zcu104/reports/hyperstarcop_arm_results_threads2/benchmark_summary.csv).

## 8. Como FPS e latência foram calculados

O throughput usado em todas as comparações é:

```text
throughput_fps = itens concluídos / tempo total de parede em segundos
```

A latência é medida individualmente:

```text
latência = instante de conclusão − instante de chegada ao pipeline
```

No `model-only`, a entrada já está carregada e normalizada e somente a inferência é medida. No `end-to-end`, entram leitura dos quatro TIFFs, normalização, filas/espera por slot, inferência, sigmoid e limiarização. Leitura do rótulo, métricas de qualidade e escrita dos resultados ficam fora da região temporizada.

Em execução sequencial, `FPS ≈ 1000 / latência_ms`. Em uma configuração concorrente isso deixa de ser verdade para uma imagem isolada, pois vários runners processam itens simultaneamente. Assim, a CPU x86 pode apresentar 5,305 FPS com latência média de 1,838 s: essa latência inclui permanência em filas, enquanto o pipeline conclui várias imagens em paralelo.

O campo `equiv_fps` dos CSVs é o inverso da latência de uma requisição. Para comparar capacidade sustentada deve-se usar `throughput_fps`.

## 9. Comparações diretas e interpretação

### 9.1 Mesmo baseline de pipeline

| Modo | CPU x86 FP32 | DPU INT8 | Vantagem da DPU | Redução de latência na DPU |
|---|---:|---:|---:|---:|
| `model-only` | 2,072 FPS | 26,472 FPS | 12,773× | 92,17% |
| `end-to-end` | 2,010 FPS | 4,821 FPS | 2,399× | 58,29% |

### 9.2 CPU ARM conforme o número de threads

| Comparação de FPS | 1 thread ARM | 2 threads ARM | 4 threads ARM |
|---|---:|---:|---:|
| CPU x86 baseline / ARM, `model-only` | 12,960× | 7,089× | 4,174× |
| CPU x86 baseline / ARM, `end-to-end` | 12,851× | 7,159× | 4,332× |
| DPU baseline / ARM, `model-only` | 165,545× | 90,551× | 53,312× |
| DPU baseline / ARM, `end-to-end` | 30,828× | 17,175× | 10,392× |

O custo de E2E é proporcionalmente muito maior na DPU do que na CPU ARM: a inferência da DPU é curta, então leitura, normalização e pós-processamento passam a dominar. Na CPU ARM, a inferência FP32 domina o tempo total, e o acréscimo do pré/pós-processamento fica próximo de 140 ms.

## 10. Limitações e regras para citar os números

- O `STARCOP_mini` contém somente nove imagens de teste com pluma. Ele permite reprodução funcional, mas não substitui uma avaliação estatística no conjunto completo.
- CPU/ARM usam FP32 e a DPU usa INT8. A tabela de qualidade demonstra que a quantização preservou F1/IoU neste conjunto, mas as precisões numéricas não são idênticas.
- Os benchmarks da CPU ARM possuem menos amostras que os baselines x86/DPU. Eles são estáveis entre as repetições existentes, mas uma campanha longa ainda seria desejável para publicação.
- O melhor `model-only` da DPU foi medido uma vez com 90 itens; deve ser citado dessa forma. Não se deve descrevê-lo como média de três execuções.
- Resultados de busca com múltiplos runners devem ser comparados por `throughput_fps`; o inverso da latência não representa a vazão total nesses casos.
- Os resultados do snapshot CPU `optimized` são históricos. Para números finais da CPU x86, devem ser usados `best_configs.json` e as três repetições da fase `50_final`.
- Dois núcleos Ryzen, duas threads XNNPACK e dois núcleos DPU não são arquiteturalmente equivalentes; a contagem serve apenas como referência de recursos usados.

## 11. Fontes auditadas

| Conjunto | Arquivo principal |
|---|---|
| Relatório anterior CPU × DPU | [`relatorio.md`](relatorio.md) |
| Baseline e qualidade DPU | [`hyperstarcop_cpp_results`](zcu104/hyperstarcop_cpp_results/benchmark_summary.csv) |
| Busca DPU, 47 execuções | [`all_runs.csv`](hyperstarcop_sweep_results/all_runs.csv) |
| Configuração DPU selecionada | [`best_config.json`](hyperstarcop_sweep_results/best_config.json) |
| Reprodução da melhor DPU E2E | [`best_reproduction`](hyperstarcop_sweep_results/best_reproduction/benchmark_summary.csv) |
| Comparação CPU nas configurações DPU | [`benchmark_summary_comparavel_zcu104.csv`](../resultados_cpu/hyperstarcop_cpu_zcu104_comparison/benchmark_summary_comparavel_zcu104.csv) |
| Busca CPU, 37 execuções | [`all_runs.csv`](../resultados_cpu/hyperstarcop_cpu_sweep/all_runs.csv) |
| Melhores configurações CPU | [`best_configs.json`](../resultados_cpu/hyperstarcop_cpu_sweep/best_configs.json) |
| Snapshot CPU anterior | [`hyperstarcop_cpu_optimized`](../resultados_cpu/hyperstarcop_cpu_optimized/benchmark_summary.csv) |
| Exportação PTE/XNNPACK | [`export_xnnpack.json`](../Arm_zcu104/reports/export_xnnpack.json) |
| Validação PyTorch × PTE | [`comparacao_float_vs_pte.csv`](../Arm_zcu104/reports/comparacao_float_vs_pte.csv) |
| ARM 1 thread | [`benchmark_summary.csv`](../Arm_zcu104/reports/hyperstarcop_arm_results_threads1/benchmark_summary.csv) |
| ARM 2 threads e qualidade | [`benchmark_summary.csv`](../Arm_zcu104/reports/hyperstarcop_arm_results_threads2/benchmark_summary.csv) |
| ARM 1/4 threads adicionais | [`benchmark_summary.csv`](../Arm_zcu104/reports/hyperstarcop_arm_results_threads4/benchmark_summary.csv) |

## 12. Conclusão

A DPU obteve a maior vazão em todos os cenários: 50,876 FPS `model-only` e média de 21,673 FPS `end-to-end`. A CPU x86 chegou a 6,076 e 5,305 FPS após busca de paralelismo. A CPU ARM executou corretamente o mesmo FP32 via ExecuTorch/XNNPACK, preservou exatamente as máscaras e escalou até 0,497 e 0,464 FPS com quatro threads, mas permaneceu limitada pelo custo da inferência FP32 no Cortex-A53.

Para comparação reprodutível, os números mais seguros são os baselines sequenciais. Para capacidade máxima, devem ser usadas as médias finais da CPU x86 e da DPU, acompanhadas das configurações e contagens de repetição apresentadas neste relatório.
