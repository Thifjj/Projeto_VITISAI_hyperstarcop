# Resultados da ZCU104

## Comece por aqui

- [comparacao_benchmarks_configuracoes.md](comparacao_benchmarks_configuracoes.md): comparação completa entre CPU x86, CPU ARM e DPU;
- [relatorio.md](relatorio.md): análise detalhada da referência FP32 e da DPU INT8.

## Dados da DPU

| Caminho | Conteúdo |
|---|---|
| `zcu104/hyperstarcop_cpp_results/` | baseline, qualidade INT8 e informações do alvo |
| `hyperstarcop_sweep_results/all_runs.csv` | 47 execuções da busca |
| `hyperstarcop_sweep_results/ranking_final_runs.csv` | candidatas finais |
| `hyperstarcop_sweep_results/best_config.json` | configuração selecionada |
| `hyperstarcop_sweep_results/best_reproduction/` | reprodução adicional da vencedora |

Resultados principais:

| Modo | Baseline | Melhor |
|---|---:|---:|
| `model-only` | 26,472 FPS | 50,876 FPS |
| `end-to-end` | 4,821 FPS | 21,673 FPS |

Qualidade INT8: precisão `0,909764`, recall `0,903945`, F1 `0,906845`, IoU `0,829567` e acurácia `0,996554`.

Para executar novamente, consulte [to_zcu/README.md](../to_zcu/README.md).
