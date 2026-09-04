# Fluxo Vitis AI

Esta pasta reúne a transformação do checkpoint original em uma rede FP32 independente, sua quantização INT8 e o XModel da ZCU104.

## Artefatos

| Caminho | Conteúdo |
|---|---|
| `float_model/hyperstarcop_network_fp32.pth` | `state_dict` da U-Net independente |
| `inspect/` | relatório de compatibilidade com a DPU |
| `quantized_int8_calib200/` | configuração e artefatos INT8 |
| `compiled/zcu104_b4096/hyperstarcop.xmodel` | modelo implantado na placa |
| `results_modelo_puro/` | equivalência checkpoint × rede extraída |
| `results_int8_calib200/` | qualidade FP32 × INT8 |

## 1. Extrair e validar a rede FP32

No ambiente principal:

```bash
source venv/bin/activate
python vitis_ai/scripts/01_comparar_modelo_original_rede_pura.py
python vitis_ai/scripts/02_exportar_rede_pura.py
python vitis_ai/scripts/03_validar_rede_standalone.py
```

Esses passos geram e validam `float_model/hyperstarcop_network_fp32.pth`.

## 2. Inspecionar e quantizar

Execute dentro do ambiente/container Vitis AI compatível com `pytorch_nndct`:

```bash
python vitis_ai/scripts/04_inspecionar_vitis_ai.py

python vitis_ai/scripts/05_quantizar_int8.py \
  --quant_mode calib \
  --calib_samples 200 \
  --dataset_root /workspace/dataset_STARCOP

python vitis_ai/scripts/05_quantizar_int8.py \
  --quant_mode test \
  --calib_samples 200 \
  --dataset_root /workspace/dataset_STARCOP \
  --test_root STARCOP_mini \
  --deploy
```

O dataset indicado por `--dataset_root` deve ser o **STARCOP completo** e conter `train.csv`. A calibração validada usou 200 imagens. O `STARCOP_mini` é usado apenas no teste posterior.

O `setup_environment.sh` não instala Vitis AI nem `pytorch_nndct`.

## 3. Implantar

O artefato usado pelo runner está em:

```text
compiled/zcu104_b4096/hyperstarcop.xmodel
```

Alvo: `DPUCZDX8G_ISA1_B4096`, entrada NHWC `[1, 512, 512, 4]`, saída NHWC `[1, 512, 512, 1]`.

Para execução e benchmark, siga [to_zcu/README.md](../to_zcu/README.md).

## Ordem dos scripts

Os nomes `01` a `05` representam dependências reais; não execute a quantização antes de gerar e validar o modelo em `float_model/`.
