# Resultados da CPU x86

| Diretório | Conteúdo | Uso recomendado |
|---|---|---|
| `hyperstarcop_cpu_zcu104_comparison/` | baseline e reprodução das configurações da DPU | comparação direta CPU × DPU |
| `hyperstarcop_cpu_sweep/` | 37 execuções de busca e confirmações finais | melhores resultados da CPU |
| `hyperstarcop_cpu_optimized/` | snapshot anterior da implementação | histórico |

Arquivos principais da busca:

- `hyperstarcop_cpu_sweep/all_runs.csv`: todas as configurações;
- `hyperstarcop_cpu_sweep/final_runs.csv`: seis confirmações de 500 itens;
- `hyperstarcop_cpu_sweep/best_configs.json`: vencedoras por modo.

Resultados finais:

| Modo | Configuração | FPS médio |
|---|---|---:|
| `model-only` | 4 runners × 2 threads, 8 núcleos | 6,076 |
| `end-to-end` | 8 runners × 1 thread, Pré2, Pós1, 8 slots | 5,305 |

Use os valores de `best_configs.json` como resultados oficiais da CPU. O relatório completo está em [resultados_zcu104/comparacao_benchmarks_configuracoes.md](../resultados_zcu104/comparacao_benchmarks_configuracoes.md).
