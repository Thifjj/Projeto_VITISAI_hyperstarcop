#!/usr/bin/env bash

set -Eeuo pipefail

# Prepara dois ambientes independentes:
#   1. venv: projeto e benchmarks atuais;
#   2. venv_executorch: exportação/validação ExecuTorch + XNNPACK.
# Também baixa e valida o STARCOP_mini.
#
# O segundo ambiente é usado no notebook para gerar o arquivo .pte. Ele não
# instala o runtime C++ AArch64 na ZCU104 e não depende do Python 3.9.9 da placa.
#
# Uso:
#   ./setup_environment.sh                   # prepara os dois ambientes
#   ./setup_environment.sh --main-only       # prepara somente venv/
#   ./setup_environment.sh --executorch-only # prepara só venv_executorch/
#   ./setup_environment.sh --check           # verifica ambos, sem alterar

readonly REQUIRED_PYTHON_VERSION="3.10.20"
readonly PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly VENV_DIR="$PROJECT_ROOT/venv"
readonly VENV_PYTHON="$VENV_DIR/bin/python"
readonly REQUIREMENTS_FILE="$PROJECT_ROOT/requirements-environment.txt"
readonly EXECUTORCH_VENV_DIR="$PROJECT_ROOT/venv_executorch"
readonly EXECUTORCH_PYTHON="$EXECUTORCH_VENV_DIR/bin/python"
readonly EXECUTORCH_REQUIREMENTS_FILE="$PROJECT_ROOT/requirements-executorch.txt"
readonly DATASET_DIR="$PROJECT_ROOT/STARCOP_mini"
readonly DATASET_ZIP="$PROJECT_ROOT/STARCOP_mini.zip"
readonly DATASET_GDRIVE_ID="1Qw96Drmk2jzBYSED0YPEUyuc2DnBechl"
readonly PYTHON_BIN="${PYTHON_BIN:-python3.10}"

MODE="all"
CHECK_ONLY="false"
for argument in "$@"; do
    case "$argument" in
        --main-only) MODE="main" ;;
        --executorch-only) MODE="executorch" ;;
        --check) CHECK_ONLY="true" ;;
        -h|--help)
            printf '%s\n' \
                "Uso: $0 [OPÇÃO]" \
                "" \
                "Sem opção: prepara venv/ e venv_executorch/." \
                "  --main-only        seleciona somente o ambiente principal" \
                "  --executorch-only  seleciona somente o ambiente ExecuTorch" \
                "  --check            apenas verifica o ambiente selecionado" \
                "" \
                "Exemplo: $0 --main-only --check"
            exit 0
            ;;
        *)
            echo "Opção desconhecida: $argument" >&2
            echo "Use '$0 --help'." >&2
            exit 2
            ;;
    esac
done

info() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
fail() { printf '[ERRO] %s\n' "$*" >&2; exit 1; }

python_version() {
    "$1" -c 'import platform; print(platform.python_version())'
}

require_python_31020() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail \
        "'$PYTHON_BIN' não foi encontrado. Instale o Python $REQUIRED_PYTHON_VERSION ou informe PYTHON_BIN."

    local detected
    detected="$(python_version "$PYTHON_BIN")"
    [[ "$detected" == "$REQUIRED_PYTHON_VERSION" ]] || fail \
        "Python incompatível em '$PYTHON_BIN': $detected (exigido: $REQUIRED_PYTHON_VERSION)."
    ok "Python base: $detected ($PYTHON_BIN)"
}

validate_venv() {
    [[ -x "$VENV_PYTHON" ]] || fail "venv ausente ou inválido em: $VENV_DIR"

    local detected
    detected="$(python_version "$VENV_PYTHON")"
    [[ "$detected" == "$REQUIRED_PYTHON_VERSION" ]] || fail \
        "O venv usa Python $detected; é necessário recriá-lo com Python $REQUIRED_PYTHON_VERSION."
    ok "venv usa Python $detected"
}

validate_pinned_packages() {
    local environment_python="$1"
    local requirements_file="$2"

    "$environment_python" - "$requirements_file" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

requirements = Path(sys.argv[1])
if not requirements.is_file():
    raise SystemExit(f"Arquivo de dependências ausente: {requirements}")

problems = []
for raw_line in requirements.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if " @ " in line:
        package = line.split(" @ ", 1)[0].strip()
        try:
            version(package)
        except PackageNotFoundError:
            problems.append(f"{package}: não instalado")
        continue
    elif "==" not in line:
        problems.append(f"requisito sem versão fixa: {line}")
        continue
    package, expected = line.split("==", 1)
    try:
        installed = version(package)
    except PackageNotFoundError:
        problems.append(f"{package}: não instalado")
    else:
        if installed != expected:
            problems.append(f"{package}: instalado {installed}, esperado {expected}")

if problems:
    raise SystemExit("Dependências divergentes:\n- " + "\n- ".join(problems))
print("Dependências fixadas conferidas.")
PY
    "$environment_python" -m pip check
    ok "dependências Python válidas"
}

prepare_venv() {
    local environment_dir="$1"
    local environment_python="$2"
    local requirements_file="$3"
    local label="$4"

    [[ -f "$requirements_file" ]] || fail "Arquivo ausente: $requirements_file"

    if [[ ! -x "$environment_python" ]]; then
        info "Criando $label com Python $REQUIRED_PYTHON_VERSION..."
        "$PYTHON_BIN" -m venv "$environment_dir"
    else
        local detected
        detected="$(python_version "$environment_python")"
        [[ "$detected" == "$REQUIRED_PYTHON_VERSION" ]] || fail \
            "$label usa Python $detected; esperado: $REQUIRED_PYTHON_VERSION."
        info "Atualizando scripts de ativação de $label..."
        "$PYTHON_BIN" -m venv --upgrade "$environment_dir"
    fi

    info "Atualizando ferramentas de instalação em $label..."
    "$environment_python" -m pip install --upgrade pip setuptools wheel

    info "Instalando dependências fixadas de $label..."
    "$environment_python" -m pip install --requirement "$requirements_file"
    validate_pinned_packages "$environment_python" "$requirements_file"
    ok "$label preparado"
}

validate_executorch() {
    [[ -x "$EXECUTORCH_PYTHON" ]] || fail \
        "venv ExecuTorch ausente em: $EXECUTORCH_VENV_DIR"

    local detected
    detected="$(python_version "$EXECUTORCH_PYTHON")"
    [[ "$detected" == "$REQUIRED_PYTHON_VERSION" ]] || fail \
        "venv ExecuTorch usa Python $detected; esperado: $REQUIRED_PYTHON_VERSION."

    validate_pinned_packages \
        "$EXECUTORCH_PYTHON" \
        "$EXECUTORCH_REQUIREMENTS_FILE"

    "$EXECUTORCH_PYTHON" - <<'PY'
import executorch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
print("ExecuTorch e particionador XNNPACK importados com sucesso.")
PY
    ok "ambiente ExecuTorch/XNNPACK válido"
}

validate_dataset() {
    "$VENV_PYTHON" - "$DATASET_DIR" <<'PY'
import csv
from pathlib import Path
import sys

root = Path(sys.argv[1])
csv_names = ("train_mini10.csv", "test_mini10.csv")
required_files = (
    "mag1c.tif",
    "TOA_AVIRIS_640nm.tif",
    "TOA_AVIRIS_550nm.tif",
    "TOA_AVIRIS_460nm.tif",
    "labelbinary.tif",
)

problems = []
sample_count = 0
for csv_name in csv_names:
    csv_path = root / csv_name
    if not csv_path.is_file():
        problems.append(f"arquivo ausente: {csv_name}")
        continue
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 9:
        problems.append(f"{csv_name}: {len(rows)} amostras (esperadas: 9)")
    sample_count += len(rows)
    for row in rows:
        sample_id = row.get("id") or row.get("name") or next(iter(row.values()), "")
        sample_dir = root / sample_id
        if not sample_id:
            problems.append(f"{csv_name}: linha sem identificador")
            continue
        for filename in required_files:
            if not (sample_dir / filename).is_file():
                problems.append(f"arquivo ausente: {sample_id}/{filename}")

if problems:
    preview = problems[:20]
    suffix = f"\n... e mais {len(problems) - 20}" if len(problems) > 20 else ""
    raise SystemExit("STARCOP_mini inválido:\n- " + "\n- ".join(preview) + suffix)
print(f"STARCOP_mini válido: {sample_count} entradas (9 treino + 9 teste).")
PY
    ok "dataset STARCOP_mini válido"
}

download_dataset() {
    [[ ! -e "$DATASET_DIR" ]] || fail \
        "O diretório STARCOP_mini existe, mas está incompleto. Mova-o ou corrija-o antes de executar novamente."

    if [[ ! -f "$DATASET_ZIP" ]]; then
        info "Baixando STARCOP_mini..."
        "$VENV_PYTHON" - "$DATASET_GDRIVE_ID" "$DATASET_ZIP" <<'PY'
import sys
import gdown

file_id, output = sys.argv[1:]
result = gdown.download(id=file_id, output=output, quiet=False)
if not result:
    raise SystemExit("Falha ao baixar o STARCOP_mini do Google Drive.")
PY
    else
        info "Usando arquivo existente: $DATASET_ZIP"
    fi

    info "Extraindo STARCOP_mini..."
    "$VENV_PYTHON" - "$DATASET_ZIP" "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import sys
import zipfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive) as zipped:
    for member in zipped.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise SystemExit(f"Caminho inseguro no ZIP: {member.filename}")
    zipped.extractall(destination)
PY
}

printf '\nPreparação do ambiente HyperSTARCOP\nProjeto: %s\n\n' "$PROJECT_ROOT"
require_python_31020

if [[ "$CHECK_ONLY" == "true" ]]; then
    if [[ "$MODE" == "all" || "$MODE" == "main" ]]; then
        validate_venv
        validate_pinned_packages "$VENV_PYTHON" "$REQUIREMENTS_FILE"
        validate_dataset
    fi
    if [[ "$MODE" == "all" || "$MODE" == "executorch" ]]; then
        validate_executorch
    fi
    printf '\nAmbiente conferido; nenhuma alteração foi feita.\n'
    exit 0
fi

if [[ "$MODE" == "all" || "$MODE" == "main" ]]; then
    prepare_venv "$VENV_DIR" "$VENV_PYTHON" "$REQUIREMENTS_FILE" "venv principal"

    if ! validate_dataset; then
        download_dataset
        validate_dataset
    fi
fi

if [[ "$MODE" == "all" || "$MODE" == "executorch" ]]; then
    prepare_venv \
        "$EXECUTORCH_VENV_DIR" \
        "$EXECUTORCH_PYTHON" \
        "$EXECUTORCH_REQUIREMENTS_FILE" \
        "venv ExecuTorch/XNNPACK"
    validate_executorch
fi

printf '\nAmbientes prontos.\n'
if [[ "$MODE" == "all" || "$MODE" == "main" ]]; then
    printf 'Projeto atual:\n  source "%s/bin/activate"\n' "$VENV_DIR"
fi
if [[ "$MODE" == "all" || "$MODE" == "executorch" ]]; then
    printf 'Exportação ExecuTorch/XNNPACK:\n  source "%s/bin/activate"\n' "$EXECUTORCH_VENV_DIR"
    printf 'Observação: o runtime C++ AArch64 da ZCU104 deve ser compilado separadamente.\n'
fi
