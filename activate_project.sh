#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$PROJECT_ROOT/venv/bin/activate"

export PYTHONPATH="/media/jacques/hdd/Laboratorio/projeto_metano/Projeto_VITISAI_hyperstarcop/STARCOP:${PYTHONPATH:-}"

echo "Ambiente Projeto_Metano ativado."
echo "Python: $(which python3)"
echo "PYTHONPATH: /media/jacques/hdd/Laboratorio/projeto_metano/Projeto_VITISAI_hyperstarcop/STARCOP"
