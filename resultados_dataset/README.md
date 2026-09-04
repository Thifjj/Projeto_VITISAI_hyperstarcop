# Referência de qualidade FP32

Este diretório contém a validação do modelo PyTorch FP32 nas nove imagens de teste do `STARCOP_mini`.

| Arquivo | Conteúdo |
|---|---|
| `metricas_globais.csv` | TP, FP, FN, TN, precisão, recall, F1, IoU e acurácia globais |
| `metricas_por_imagem.csv` | métricas de cada imagem |
| `validacao_*.png` | visualizações geradas pela validação |
| `uma_imagem/` | saída opcional do teste visual individual |

Referência global: precisão `0,892663`, recall `0,920803`, F1 `0,906515`, IoU `0,829014` e acurácia `0,996476`.

Para regenerar:

```bash
source venv/bin/activate
python scripts/validando_rede_minidataset.py
```
