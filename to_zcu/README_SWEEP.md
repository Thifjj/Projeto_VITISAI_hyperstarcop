# Sweep de desempenho do HyperSTARCOP na ZCU104

O `sweep_hyperstarcop_zcu104.cpp` procura a melhor configuração do benchmark
C++ sem executar o produto cartesiano completo de todos os parâmetros. Tanto
o benchmark quanto o controlador do sweep são executáveis C++17. Não existe
dependência de Python.

## Por que o limite padrão é 16

A ZCU104 analisada possui dois núcleos DPU B4096 a 300 MHz. O sweep permite
até 16 workers e até 16 slots totais em voo, mas limita runners VART a quatro
por padrão: dois runners por núcleo DPU. Acima disso cresce o consumo de
memória dos contextos do modelo e normalmente aumenta contenção, sem criar
novos núcleos de execução.

Esse limite pode ser alterado explicitamente com `--max-runners`, sempre até
16, para investigar uma imagem da placa diferente. A configuração padrão é a
mais segura para esta ZCU104.

## Preparação na placa

Copie para a mesma pasta:

- `hyperstarcop_zcu104_optimized.cpp`;
- `build_hyperstarcop_zcu104_optimized.sh`;
- `sweep_hyperstarcop_zcu104.cpp`.

Compile e confira a campanha sem executá-la:

```bash
chmod +x build_hyperstarcop_zcu104_optimized.sh
./build_hyperstarcop_zcu104_optimized.sh

./sweep_hyperstarcop_zcu104 --dry-run
```

## Execução recomendada

Com os caminhos padrão em `/home/root`:

```bash
./sweep_hyperstarcop_zcu104 --resume
```

Com caminhos explícitos:

```bash
./sweep_hyperstarcop_zcu104 \
  --binary ./hyperstarcop_zcu104_optimized \
  --model /home/root/hyperstarcop.xmodel \
  --dataset /home/root/STARCOP_mini \
  --csv /home/root/STARCOP_mini/test_mini10.csv \
  --out /home/root/hyperstarcop_sweep_results \
  --max-concurrency 16 \
  --search-iterations 90 \
  --final-iterations 500 \
  --final-candidates 3 \
  --final-repeats 3 \
  --pin-modes both \
  --resume
```

`--resume` reutiliza somente execuções completas com a mesma identidade,
incluindo a quantidade de iterações. Uma campanha interrompida pode ser
iniciada novamente com o mesmo comando.

## Etapas da busca

1. Executa baseline e validação de qualidade uma única vez.
2. Mede de 1 a 4 runners no modo model-only, com e sem afinidade.
3. Leva os dois melhores candidatos para o pipeline end-to-end.
4. Testa `1, 2, 4, 8, 16` workers de pré-processamento.
5. Testa `1, 2, 4, 8, 16` workers de pós-processamento.
6. Aumenta a quantidade de slots, mantendo no máximo 16 no total.
7. Combina os dois melhores valores de cada eixo em até 16 testes de
   refinamento, capturando interações que uma busca eixo a eixo poderia perder.
8. Repete três vezes, com 500 imagens, os três melhores candidatos.

Os testes curtos servem somente para busca. A melhor configuração é escolhida
pela mediana do throughput nas confirmações longas. P99 de latência é usado
como desempate.

## Resultados

O diretório de saída contém:

- `campaign.txt`: limites e parâmetros da campanha;
- `runs/`: CSV e log completos de cada execução;
- `all_runs.csv`: todas as medições agregadas;
- `ranking_search.csv`: classificação dos testes curtos end-to-end;
- `ranking_final_runs.csv`: classificação das repetições longas;
- `best_config.json`: configuração selecionada e suas medianas;
- `reproduce_best.sh`: comando pronto para reproduzir o melhor resultado.

Cada execução usa um diretório próprio, portanto métricas e CSVs não são
sobrescritos por outra configuração.

## Relação com o benchmark de CPU

Os nomes representam os mesmos estágios usados no benchmark CPU:

| ZCU104 | CPU | Significado |
|---|---|---|
| `runners` | `workers` | executores concorrentes da rede |
| `pre_workers` | `pre_workers` | leitura e normalização |
| `post_workers` | `post_workers` | sigmoid e limiarização |
| `total_slots` | `slots` | amostras máximas em voo |

No código da placa, `slots-per-runner` é local a cada runner. Por isso o sweep
também registra `total_slots = runners × slots_per_runner`. Esse é o valor que
deve ser comparado futuramente com `--slots` do benchmark CPU.

O batch permanece 1 em ambas as plataformas. O critério principal é
`throughput_fps`; `latency_p99_ms`, tempo de DPU, pré-processamento,
pós-processamento e I/O ajudam a identificar o gargalo.
