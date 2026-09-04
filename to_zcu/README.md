# Execução do HyperSTARCOP na DPU da ZCU104

Este diretório contém o runner C++/VART e o controlador da busca de desempenho. Ambos são compilados e executados diretamente na ZCU104; não dependem de Python.

## Arquivos

| Arquivo | Função |
|---|---|
| `hyperstarcop_zcu104_optimized.cpp` | valida qualidade e mede `model-only`/`end-to-end` |
| `sweep_hyperstarcop_zcu104.cpp` | testa runners, workers, slots e afinidade |
| `build_hyperstarcop_zcu104_optimized.sh` | compila os dois executáveis |

## Preparação

Copie este diretório, o XModel e o dataset para a placa. Exemplo no notebook:

```bash
scp -O -r to_zcu root@IP_DA_PLACA:/home/root/
scp -O vitis_ai/compiled/zcu104_b4096/hyperstarcop.xmodel \
  root@IP_DA_PLACA:/home/root/
scp -O -r STARCOP_mini root@IP_DA_PLACA:/home/root/
```

Na placa:

```bash
cd /home/root/to_zcu
chmod +x build_hyperstarcop_zcu104_optimized.sh
./build_hyperstarcop_zcu104_optimized.sh
```

São necessários GCC/G++ 11, OpenCV, headers VART/XIR e `pkg-config`, já disponíveis na imagem usada neste projeto.

## Benchmark básico

```bash
./hyperstarcop_zcu104_optimized \
  --profile all \
  --model /home/root/hyperstarcop.xmodel \
  --dataset /home/root/STARCOP_mini \
  --csv /home/root/STARCOP_mini/test_mini10.csv \
  --out /home/root/hyperstarcop_cpp_results
```

Perfis:

- `baseline`: uma imagem em voo;
- `max-model-only`: busca vazão de inferência com entradas prontas;
- `max-e2e`: mede o pipeline completo;
- `all`: valida qualidade e executa os perfis configurados.

Use `./hyperstarcop_zcu104_optimized --help` para ver todos os parâmetros.

## Busca automática

Confira a campanha sem executar:

```bash
./sweep_hyperstarcop_zcu104 --dry-run
```

Execução recomendada:

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

`--resume` reaproveita somente execuções completas com a mesma configuração e quantidade de iterações.

## Estratégia da busca

1. valida o baseline;
2. testa de 1 a 4 runners em `model-only`, com e sem afinidade;
3. testa os melhores runners no pipeline E2E;
4. varia workers de pré-processamento;
5. varia workers de pós-processamento;
6. varia slots por runner, mantendo até 16 itens em voo;
7. refina as melhores combinações;
8. confirma três candidatas com 3 × 500 itens.

A melhor configuração é escolhida pela mediana de `throughput_fps`; P99 é o desempate.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `runs/` | logs e CSVs de cada configuração |
| `all_runs.csv` | todas as execuções agregadas |
| `ranking_search.csv` | busca curta ordenada |
| `ranking_final_runs.csv` | repetições finais ordenadas |
| `best_config.json` | configuração selecionada |
| `reproduce_best.sh` | comando de reprodução |

Cada execução possui um diretório próprio e não sobrescreve outra configuração.

## Resultado selecionado

| Modo | Configuração | Resultado |
|---|---|---:|
| `model-only` | 4 runners, 1 slot/runner, sem pin | 50,876 FPS |
| `end-to-end` | 3 runners, Pré4, Pós16, 5 slots/runner, pin | 21,673 FPS em média |

O resultado `model-only` é uma busca de 90 itens. O E2E é a média das três execuções finais de 500 itens.

Dados completos: [resultados_zcu104/comparacao_benchmarks_configuracoes.md](../resultados_zcu104/comparacao_benchmarks_configuracoes.md).
