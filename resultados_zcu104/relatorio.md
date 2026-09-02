# Relatório consolidado dos resultados na ZCU104

## 1. Escopo

Este relatório compara a execução do HyperSTARCOP em duas plataformas:

- CPU do notebook com o modelo original em FP32;
- DPU da Xilinx ZCU104 com o modelo quantizado em INT8.

A comparação usa o mesmo modelo lógico, o mesmo conjunto de 9 imagens, `batch = 1`, o mesmo pré-processamento e o mesmo limiar de segmentação (`sigmoid > 0,5`). Para o desempenho foram considerados somente os seguintes cenários:

- baseline, com 1 runner, 1 worker de pré-processamento, 1 worker de pós-processamento e 1 slot;
- melhor configuração `model-only` encontrada na ZCU104: 4 runners e 1 slot por runner;
- melhor configuração `end-to-end` encontrada na ZCU104: 3 runners, 4 workers de pré-processamento, 16 workers de pós-processamento e 5 slots por runner, totalizando 15 slots.
- execução limitada a 2 núcleos físicos distintos da CPU, para comparação com os 2 núcleos DPU da ZCU104;
- busca da vazão máxima específica da CPU, usando até os 8 núcleos físicos e 16 threads lógicas do notebook.

Os valores FP32 foram medidos novamente no notebook com o perfil `zcu104-comparison`. Os resultados estão em `resultados_cpu/hyperstarcop_cpu_zcu104_comparison`. Os valores INT8 vêm dos CSVs em `resultados_zcu104`.

## 2. Resumo em tabela metricas mesuradas em FP32 (Modelo original)

| Indicador FP32 na CPU | Resultado |
|---|---:|
| Imagens avaliadas | 9 |
| Verdadeiros positivos (TP) | 40.310 |
| Falsos positivos (FP) | 4.847 |
| Falsos negativos (FN) | 3.467 |
| Verdadeiros negativos (TN) | 2.310.672 |
| Precisão global | 0,892663 (89,2663%) |
| Recall global | 0,920803 (92,0803%) |
| F1 global | 0,906515 (90,6515%) |
| IoU global | 0,829014 (82,9014%) |
| Acurácia global | 0,996476 (99,6476%) |
| Baseline `model-only` | 2,072 FPS; latência média de 482,515 ms |
| Baseline `end-to-end` | 2,010 FPS; latência média de 497,361 ms |
| Configuração equivalente à melhor `model-only` da ZCU104 | 5,278 FPS; latência média de 749,623 ms |
| Configuração equivalente à melhor `end-to-end` da ZCU104 | 4,050 FPS; latência média de 4.587,699 ms |
| Melhor `model-only` específico da CPU (média de 3 × 500) | 6,076 FPS; latência média de 657,373 ms |
| Melhor `end-to-end` específico da CPU (média de 3 × 500) | 5,305 FPS; latência média de 1.837,972 ms |

## 3. Ambiente de execução

| Item | CPU — FP32 | ZCU104 — INT8 |
|---|---|---|
| Plataforma | AMD Ryzen 9 5980HX, 8 núcleos físicos e 16 CPUs lógicas | Xilinx ZCU104, AArch64 |
| Sistema | Linux 7.0.0-30-generic | PetaLinux 2022.2, kernel 5.15.36-xilinx-v2022.2 |
| Framework/runtime  | Python 3.10.2 / PyTorch 2.13.0+cu130 | Componentes Vitis AI/VART 3.5.0 |
| Precisão | FP32 | INT8 |
| Batch | 1 | 1 |
| Paralelismo interno | 1 thread intra-op e 1 inter-op | 2 núcleos DPUCZDX8G_ISA1_B4096 |
| Frequência do acelerador | Não se aplica | 300 MHz |
| Entrada do modelo | 1 × 512 × 512 × 4 | 1 × 512 × 512 × 4 |
| Saída do modelo | 1 × 512 × 512 × 1 | 1 × 512 × 512 × 1 |
| Parâmetros do modelo | 6.629.233 | Mesmo modelo lógico quantizado |

Na CPU, cada runner é uma cópia independente do modelo FP32. Na ZCU104, cada runner é uma instância de execução VART associada ao modelo compilado para a DPU.

## 4. Qualidade da segmentação

### 4.1 Métricas globais

Resultados globais medidos na ZCU104 com o modelo INT8:

| Imagens | TP | FP | FN | TN | Precisão | Recall | F1 | IoU | Acurácia |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 39.572 | 3.925 | 4.205 | 2.311.594 | 0,909764 | 0,903945 | 0,906845 | 0,829567 | 0,996554 |

As médias simples por imagem na ZCU104 foram: precisão 0,859369; recall 0,842942; F1 0,843366; IoU 0,739378; acurácia 0,996554.

### 4.2 Resultados por imagem FP32 (Modelo original)

Todas as 9 imagens avaliadas contêm pluma (`has_plume = True`). Os valores desta tabela foram medidos na CPU com os pesos originais FP32.

| Imagem (ID abreviado) | Pixels reais | Previstos | TP | FP | FN | TN | Precisão | Recall | F1 | IoU | Acurácia |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 144405_r2674_c436 | 3.171 | 2.737 | 2.154 | 583 | 1.017 | 258.390 | 0,7870 | 0,6793 | 0,7292 | 0,5738 | 0,9939 |
| 163108_r14851_c525 | 3.272 | 3.436 | 3.062 | 374 | 210 | 258.498 | 0,8912 | 0,9358 | 0,9129 | 0,8398 | 0,9978 |
| 141549_r3900_c244 | 360 | 44 | 44 | 0 | 316 | 261.784 | 1,0000 | 0,1222 | 0,2178 | 0,1222 | 0,9988 |
| 165545_r6573_c24 | 6.753 | 7.531 | 6.468 | 1.063 | 285 | 254.328 | 0,8589 | 0,9578 | 0,9056 | 0,8275 | 0,9949 |
| 190719_r2696_c420 | 4.100 | 3.972 | 3.455 | 517 | 645 | 257.527 | 0,8698 | 0,8427 | 0,8560 | 0,7483 | 0,9956 |
| 191828_r4300_c359 | 9.339 | 10.064 | 9.193 | 871 | 146 | 251.934 | 0,9135 | 0,9844 | 0,9476 | 0,9004 | 0,9961 |
| 181457_r4349_c389 | 1.008 | 1.410 | 980 | 430 | 28 | 260.706 | 0,6950 | 0,9722 | 0,8106 | 0,6815 | 0,9983 |
| 165503_r2660_c460 | 13.763 | 14.206 | 13.363 | 843 | 400 | 247.538 | 0,9407 | 0,9709 | 0,9556 | 0,9149 | 0,9953 |
| 190719_r1941_c33 | 2.011 | 1.757 | 1.591 | 166 | 420 | 259.967 | 0,9055 | 0,7911 | 0,8445 | 0,7308 | 0,9978 |

O melhor F1 em FP32 ocorreu em `165503_r2660_c460`, com 0,9556. O pior ocorreu em `141549_r3900_c244`, com 0,2178, principalmente pelo recall de apenas 0,1222 nessa imagem de pluma pequena.

### 4.3 Tabela de comparação qualidade da segmentacao ZCU104 (INT8) com a referência FP32 documentada

| Métrica | CPU FP32 | ZCU104 INT8 | Delta INT8 − FP32 | Interpretação |
|---|---:|---:|---:|---|
| Precisão | 0,892663 | 0,909764 | +0,017101 | INT8: +1,710 p.p. |
| Recall | 0,920803 | 0,903945 | −0,016858 | INT8: −1,686 p.p. |
| F1 | 0,906515 | 0,906845 | +0,000330 | diferença de +0,033 p.p. |
| IoU | 0,829014 | 0,829567 | +0,000552 | diferença de +0,055 p.p. |
| Acurácia | 0,996476 | 0,996554 | +0,000078 | diferença de +0,008 p.p. |

O modelo INT8 preservou a qualidade global do FP32. A quantização produziu máscaras um pouco mais conservadoras no agregado: a precisão aumentou e o recall diminuiu. F1, IoU e acurácia permaneceram praticamente inalterados.

## 5. Tabela de comparacao de benchmark CPU (FP32) x DPU ZCU104 (INT8)

### Baseline: configuração com tudo em 1

Os dois ambientes usaram 100 amostras para `model-only` e 45 amostras para `end-to-end`.

| Modo | CPU FP32 FPS | ZCU104 INT8 FPS | Aceleração da ZCU104 | CPU lat. média | ZCU104 lat. média | Redução de latência | CPU P95 | ZCU104 P95 | CPU P99 | ZCU104 P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `model-only` | 2,072 | 26,472 | 12,773× | 482,515 ms | 37,776 ms | 92,17% | 497,163 ms | 37,833 ms | 514,587 ms | 37,848 ms |
| `end-to-end` | 2,010 | 4,821 | 2,399× | 497,361 ms | 207,426 ms | 58,29% | 520,913 ms | 232,511 ms | 523,861 ms | 232,591 ms |

### Melhores configurações encontradas na ZCU104 reproduzidas na CPU

| Modo | Configuração aplicada nas duas plataformas | Amostras | CPU FP32 FPS | ZCU104 INT8 FPS | Aceleração da ZCU104 | CPU lat. média | ZCU104 lat. média | CPU P95 | ZCU104 P95 | CPU P99 | ZCU104 P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Melhor `model-only` | R4 / Pré1 / Pós1 / SPR1 / 4 slots / sem pin | 90 | 5,278 | 50,876 | 9,638× | 749,623 ms | 77,670 ms | 838,992 ms | 82,311 ms | 885,486 ms | 82,840 ms |
| Melhor `end-to-end` | R3 / Pré4 / Pós16 / SPR5 / 15 slots / com pin | 500 | 4,050 | 21,687 | 5,355× | 4.587,699 ms | 261,924 ms | 5.234,814 ms | 311,402 ms | 5.357,743 ms | 343,167 ms |

No `model-only`, a entrada já está carregada e normalizada antes da região temporizada. No `end-to-end`, a medição inclui leitura dos quatro TIFFs, normalização, inferência, sigmoid e limiarização. Métricas, rótulos e escrita dos CSVs ficam fora da região temporizada nas duas plataformas.

A latência das configurações concorrentes é medida por imagem desde a chegada ao pipeline, incluindo espera por slot e filas. Por isso, mais concorrência pode aumentar a latência individual mesmo quando melhora a vazão total. Para esses cenários, `throughput_fps` é o indicador principal da capacidade sustentada do sistema.

### Como são calculados o FPS e a latência

O FPS apresentado nas tabelas é a **vazão sustentada do sistema inteiro**, calculada pela quantidade total de imagens concluídas dividida pelo tempo de parede da execução:

```text
throughput_fps = número de imagens concluídas / tempo total medido em segundos
```

O cronômetro global começa depois que todos os workers terminam o aquecimento e estão prontos. Ele termina quando a última imagem é concluída. Aquecimento, validação das máscaras, cálculo das métricas de qualidade e gravação dos CSVs não entram nesse tempo.

A latência, por outro lado, é calculada individualmente para cada imagem:

```text
latência da imagem = instante de conclusão − instante de chegada ao pipeline
```

No `model-only`, a latência contém somente a chamada do modelo. No `end-to-end`, ela inclui espera por slot, leitura dos TIFFs, normalização, espera nas filas, inferência, sigmoid e limiarização. A média, mediana, mínimo, máximo, P90, P95 e P99 são calculados a partir dessas latências individuais.

Quando existe somente uma imagem em execução, FPS e latência são aproximadamente inversos:

```text
FPS ≈ 1000 / latência_em_ms
```

Por exemplo, no baseline CPU `model-only`, 100 imagens foram concluídas em 48,253 s:

```text
throughput = 100 / 48,253 = 2,072 FPS
FPS pela latência = 1000 / 482,515 ms = 2,072 FPS
```

Os valores coincidem porque há apenas uma inferência por vez.

Com concorrência, essa inversão direta deixa de valer, pois várias imagens avançam ao mesmo tempo. Uma aproximação útil é:

```text
throughput ≈ número médio de imagens simultâneas / latência média
```

Na repetição mediana do melhor `model-only` da CPU, 500 imagens terminaram em 82,254 s, portanto:

```text
throughput = 500 / 82,254 = 6,079 FPS
latência média = 657,020 ms = 0,657 s
4 runners / 0,657 s ≈ 6,088 FPS
```

Embora uma imagem leve aproximadamente 0,657 s, existem quatro runners processando imagens simultaneamente. Por isso o sistema entrega aproximadamente 6,08 imagens por segundo, e não apenas `1 / 0,657 = 1,52 FPS`.

No melhor `end-to-end` da CPU, a primeira repetição concluiu 500 imagens em 94,727 s:

```text
throughput = 500 / 94,727 = 5,278 FPS
latência média = 1.845,784 ms = 1,846 s
imagens simultâneas médias ≈ 5,278 × 1,846 = 9,74
```

Essa configuração possui 8 slots dentro do pipeline e 2 workers de pré-processamento que podem já ter recebido imagens enquanto aguardam um slot. Como o instante de chegada é registrado antes dessa espera, aproximadamente 9 a 10 imagens podem contribuir simultaneamente para a latência observada. Assim, é matematicamente consistente obter cerca de 5,28 FPS mesmo com latência individual próxima de 1,85 s.

Portanto, os números não significam que uma única imagem de 1,8 s equivale a 5 FPS. Eles significam que **cada imagem permanece cerca de 1,8 s no sistema, mas o pipeline conclui aproximadamente 5 imagens por segundo porque trabalha em várias imagens ao mesmo tempo**.

O CSV também contém `equiv_fps`, calculado como `1000 / latência_ms`. Esse campo representa o FPS equivalente de uma única requisição e não deve ser usado como vazão do pipeline concorrente. Para comparar a capacidade total da CPU e da ZCU104, o valor correto é sempre `throughput_fps`.

### CPU limitada a 2 núcleos físicos

O limitador de afinidade selecionou as CPUs lógicas `[0, 2]`, que pertencem a dois núcleos físicos diferentes. Foram avaliadas duas estratégias: um modelo usando duas threads internas e dois modelos concorrentes usando uma thread cada.

| Modo | Estratégia com 2 núcleos | FPS | Latência média | P95 | P99 |
|---|---|---:|---:|---:|---:|
| `model-only` | 1 runner × 2 threads | **3,770** | **265,233 ms** | **296,243 ms** | **363,273 ms** |
| `model-only` | 2 runners × 1 thread | 3,445 | 576,997 ms | 629,236 ms | 665,202 ms |
| `end-to-end` | 1 runner × 2 threads / Pré1 / Pós1 / SPR1 | **3,540** | **549,096 ms** | **590,396 ms** | **593,283 ms** |
| `end-to-end` | 2 runners × 1 thread / Pré1 / Pós1 / SPR1 | 3,477 | 833,756 ms | 1.022,007 ms | 1.071,864 ms |

Para dois núcleos, um runner com duas threads internas foi melhor nas duas modalidades. Isso indica que, nessa escala, compartilhar uma única cópia do modelo custa menos que manter duas inferências FP32 concorrentes.

A comparação “2 núcleos CPU × 2 núcleos DPU” deve ser entendida como comparação de recursos de execução, não como equivalência arquitetural: os núcleos DPU são unidades especializadas para redes neurais, enquanto os núcleos Ryzen são de propósito geral.

### Máxima vazão encontrada na CPU

A busca avaliou 37 execuções, incluindo variações de runners, threads internas, workers de pré e pós-processamento, slots, afinidade e quantidade de núcleos. As configurações vencedoras foram repetidas três vezes com 500 itens por repetição.

| Modo | Melhor configuração da CPU | FPS médio | Faixa de FPS | Latência média | P95 médio | P99 médio |
|---|---|---:|---:|---:|---:|---:|
| `model-only` | 4 runners × 2 threads; 8 núcleos físicos | **6,076** | 6,060–6,090 | 657,373 ms | 744,972 ms | 800,588 ms |
| `end-to-end` | R8 × 1 thread / Pré2 / Pós1 / SPR1 / 8 slots / 16 CPUs lógicas | **5,305** | 5,271–5,365 | 1.837,972 ms | 2.387,604 ms | 2.585,550 ms |

Em relação ao baseline CPU, a configuração `model-only` aumentou a vazão de 2,072 para 6,076 FPS, ganho de 2,932 vezes. No `end-to-end`, a vazão aumentou de 2,010 para 5,305 FPS, ganho de 2,640 vezes.

Comparando as melhores configurações específicas de cada plataforma, a ZCU104 alcançou 50,876 FPS contra 6,076 FPS da CPU em `model-only` (8,373 vezes mais). Em `end-to-end`, a média final da ZCU104 foi 21,673 FPS contra 5,305 FPS da CPU (4,086 vezes mais).

## 6. Principais conclusões

1. A validação CPU reproduziu exatamente a referência FP32 documentada: F1 de 0,906515 e IoU de 0,829014.
2. A quantização INT8 preservou a qualidade global: o F1 aumentou apenas 0,000330 e o IoU aumentou 0,000552 em relação ao FP32.
3. No baseline `model-only`, a ZCU104 foi 12,773 vezes mais rápida que a CPU do notebook e reduziu a latência média em 92,17%.
4. No baseline `end-to-end`, a ZCU104 foi 2,399 vezes mais rápida e reduziu a latência média em 58,29%.
5. Limitando o notebook a dois núcleos físicos, a melhor estratégia foi 1 runner × 2 threads: 3,770 FPS em `model-only` e 3,540 FPS em `end-to-end`.
8. Comparando as melhores configurações específicas de cada plataforma, a ZCU104 foi 8,373 vezes mais rápida em `model-only` e 4,086 vezes mais rápida em `end-to-end`.

## 7. Observações de interpretação

- Todos os testes usam `batch = 1`.
- Nos testes que reproduzem as configurações da ZCU104 na CPU, a CPU foi limitada a uma thread interna do PyTorch por runner (`intra-op = 1` e `interop-op = 1`). A busca específica da CPU também avaliou 2, 4, 8 e 16 threads internas.
- O perfil CPU reproduz os números de runners, workers, slots e afinidade das configurações vencedoras da ZCU104; ele não é uma busca independente pela melhor configuração específica do notebook.
- A seção de máxima vazão, em contraste, apresenta uma busca independente voltada à arquitetura da CPU do notebook.
- Os 2 núcleos físicos usados no teste restrito correspondem às CPUs lógicas `[0, 2]`; usar `[0, 1]` escolheria duas threads SMT do mesmo núcleo físico neste processador.
- Os valores de FPS percentil derivados de latência não representam vazão sustentada em cenários concorrentes. A vazão comparável é `throughput_fps = imagens concluídas / tempo total`.
- Os resultados da comparação direta estão em `resultados_cpu/hyperstarcop_cpu_zcu104_comparison`; as 37 execuções da busca CPU estão consolidadas em `resultados_cpu/hyperstarcop_cpu_sweep/all_runs.csv`, com as configurações finais em `best_configs.json`.
