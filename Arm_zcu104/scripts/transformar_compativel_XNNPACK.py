#!/usr/bin/env python3
"""Exporta o HyperSTARCOP FP32 para ExecuTorch com XNNPACK.

O script reconstrói a U-Net, carrega o ``state_dict`` independente e gera o
PTE estático usado pelo runner C++ AArch64 da ZCU104. Por padrão, também abre
o artefato no runtime Python local e compara sua saída com a do PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "vitis_ai/float_model/hyperstarcop_network_fp32.pth"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "Arm_zcu104/model/hyperstarcop_xnnpack_fp32.pte"
DEFAULT_REPORT = PROJECT_ROOT / "Arm_zcu104/reports/export_xnnpack.json"
DEFAULT_GRAPH = PROJECT_ROOT / "Arm_zcu104/reports/executorch_edge_graph.txt"
INPUT_SHAPE = (1, 4, 512, 512)

# O ExecuTorch chama ``flatc`` como subprocesso. Isso mantém o executável do
# ambiente ativo disponível mesmo quando o script é chamado sem ``source``.
PYTHON_BIN_DIR = str(Path(sys.executable).absolute().parent)
os.environ["PATH"] = PYTHON_BIN_DIR + os.pathsep + os.environ.get("PATH", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta o HyperSTARCOP FP32 para ExecuTorch/XNNPACK."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--graph-report", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="Gera o PTE sem abri-lo no runtime Python local.",
    )
    return parser.parse_args()


def create_model(weights_path: Path) -> torch.nn.Module:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")

    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=4,
        classes=1,
        activation=None,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 6_629_233:
        raise RuntimeError(
            f"Quantidade inesperada de parâmetros: {parameters} (esperado: 6629233)"
        )
    return model


def atomic_write_pte(program, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_bytes(program.buffer)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_with_python_runtime(
    pte_path: Path, example_input: torch.Tensor
) -> torch.Tensor:
    try:
        from executorch.runtime import Runtime
    except ImportError as error:
        raise RuntimeError(
            "Não foi possível importar o runtime Python do ExecuTorch. "
            "Confira as versões fixadas em requirements-executorch.txt."
        ) from error

    program = Runtime.get().load_program(str(pte_path))
    if "forward" not in program.method_names:
        raise RuntimeError(f"Método forward ausente no PTE: {program.method_names}")
    outputs = program.load_method("forward").execute((example_input,))
    if len(outputs) != 1:
        raise RuntimeError(f"Esperada uma saída; recebidas: {len(outputs)}")
    return outputs[0]


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = (reference - candidate).abs()
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "allclose_rtol_1e-4_atol_1e-5": bool(
            torch.allclose(reference, candidate, rtol=1e-4, atol=1e-5)
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    weights_path = args.weights.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    graph_path = args.graph_report.resolve()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    print("HyperSTARCOP -> ExecuTorch/XNNPACK")
    print("PyTorch:   ", torch.__version__)
    print("ExecuTorch:", version("executorch"))
    print("Pesos:     ", weights_path)
    print("Saída:     ", output_path)

    model = create_model(weights_path)
    example_input = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    reference_output = model(example_input)
    if tuple(reference_output.shape) != (1, 1, 512, 512):
        raise RuntimeError(f"Shape de saída inesperado: {tuple(reference_output.shape)}")
    if not torch.isfinite(reference_output).all():
        raise RuntimeError("A referência PyTorch produziu NaN ou infinito.")

    exported_program = torch.export.export(model, (example_input,), strict=True)
    edge_program = to_edge_transform_and_lower(
        exported_program, partitioner=[XnnpackPartitioner()]
    )

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph = edge_program.exported_program().graph_module.graph
    graph_path.write_text(str(graph) + "\n", encoding="utf-8")

    executorch_program = edge_program.to_executorch()
    atomic_write_pte(executorch_program, output_path)

    report = {
        "torch_version": torch.__version__,
        "executorch_version": version("executorch"),
        "segmentation_models_pytorch_version": version(
            "segmentation-models-pytorch"
        ),
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": "float32",
        "output_shape": list(reference_output.shape),
        "delegate": "XNNPACK",
        "target_runtime": "Linux AArch64 / ZCU104 Cortex-A53",
        "runtime_check": "skipped",
    }

    if not args.skip_runtime_check:
        print("Validando o PTE no runtime Python local...")
        runtime_output = run_with_python_runtime(output_path, example_input)
        errors = tensor_error(reference_output, runtime_output)
        report["runtime_check"] = "passed"
        report["comparison"] = errors
        if not errors["allclose_rtol_1e-4_atol_1e-5"]:
            raise RuntimeError("A saída XNNPACK divergiu da referência PyTorch.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("PTE gerado:", output_path)
    print("SHA-256:   ", report["output_sha256"])
    print("Relatório: ", report_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERRO: {error}", file=sys.stderr)
        raise
