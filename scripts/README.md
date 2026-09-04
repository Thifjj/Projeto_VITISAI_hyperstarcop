# Scripts do notebook

Execute estes comandos na raiz do repositório com `venv/` ativo.

## Uso normal

| Script | Uso |
|---|---|
| `test_load_checkpoint.py` | confirma que o checkpoint oficial carrega |
| `validando_1imagem_rede.py` | calcula métricas de uma imagem real |
| `validando_1imagem_plotando.py` | valida uma imagem e grava a visualização |
| `validando_rede_minidataset.py` | valida as nove imagens de teste |
| `benchmark_hyperstarcop_cpu_optimized_v2.py` | mede CPU FP32 |
| `sweep_hyperstarcop_cpu.py` | busca a maior vazão da CPU |
| `comparar_cpu_documentacao_zcu104_execucao.py` | gera comparações CPU × DPU a partir dos CSVs |

Preparação:

```bash
./setup_environment.sh --main-only
source venv/bin/activate
```

Validação recomendada:

```bash
python scripts/test_load_checkpoint.py
python scripts/validando_1imagem_rede.py
python scripts/validando_rede_minidataset.py
```

Benchmark:

```bash
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py --profile baseline
python scripts/benchmark_hyperstarcop_cpu_optimized_v2.py \
  --profile zcu104-comparison
python scripts/sweep_hyperstarcop_cpu.py
```

Use `--help` nos três scripts de benchmark/comparação para conferir opções e diretórios de saída.

## Saídas

- validação completa: `resultados_dataset/`;
- visualização de uma imagem: `resultados_dataset/uma_imagem/`;
- comparação CPU × DPU: `resultados_cpu/hyperstarcop_cpu_zcu104_comparison/`;
- busca CPU: `resultados_cpu/hyperstarcop_cpu_sweep/`.

`verificando_rede_roda.py` é um diagnóstico estrutural com tensor artificial. Ele não substitui a validação com imagens reais.
