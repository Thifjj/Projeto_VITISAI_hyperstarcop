# Relatórios ExecuTorch/XNNPACK

| Caminho | Conteúdo |
|---|---|
| `export_xnnpack.json` | versões, hashes, formas e erro numérico da exportação |
| `executorch_edge_graph.txt` | grafo Edge exportado |
| `comparacao_float_vs_pte.csv` | comparação PyTorch × PTE por imagem |
| `hyperstarcop_arm_results_threads1/` | benchmark com 1 thread |
| `hyperstarcop_arm_results_threads2/` | benchmark com 2 threads e métricas globais |
| `hyperstarcop_arm_results_threads4/` | registros adicionais de 1 e 4 threads |

Cada diretório de benchmark pode conter:

- `benchmark_samples.csv`: amostras temporais;
- `benchmark_summary.csv`: FPS, latência e percentis;
- `metricas_por_imagem.csv`: qualidade individual;
- `metricas_globais.csv`: qualidade agregada.

Resumo comparativo: [resultados_zcu104/comparacao_benchmarks_configuracoes.md](../../resultados_zcu104/comparacao_benchmarks_configuracoes.md).
