# Modelo de origem

| Arquivo | Descrição |
|---|---|
| `final_checkpoint_model.ckpt` | checkpoint oficial do HyperSTARCOP `mag1c + RGB` |
| `config.yaml` | configuração associada ao checkpoint |

SHA-256:

```text
96e274be943f64e028faded3bac3d1ee325ee7a79d6de2ee7f5deeaea1ef188d  final_checkpoint_model.ckpt
77b49dbffc61f4cbc6bea543b74ad296f60025f6b233f782ca03b1940b3d2241  config.yaml
```

O checkpoint completo é carregado pelos scripts de validação. A rede independente usada por Vitis AI e ExecuTorch é gerada em `vitis_ai/float_model/`.
