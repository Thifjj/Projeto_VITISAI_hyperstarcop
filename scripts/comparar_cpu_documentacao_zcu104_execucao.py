#!/usr/bin/env python3
"""
HyperSTARCOP - comparacao CPU x Documentacao x ZCU104

Objetivo:
  Comparar qualidade de segmentacao e desempenho, deixando explicita
  a configuracao de execucao de cada plataforma.

CONFIGURACAO ZCU104 ATUAL:
  - batch = 1
  - 1 host thread
  - 1 VART runner
  - 1 inferencia/task em voo
  - execucao sequencial
  - sem pipeline de tasks
  - sem multithreading de runners

CONFIGURACAO CPU:
  - batch = 1
  - 1 inferencia/request por vez
  - PyTorch intra-op threads: lido de system_info.json
  - PyTorch inter-op threads: lido de system_info.json

IMPORTANTE:
  PyTorch threads = 8 NAO significa 8 inferencias simultaneas.
  Significa que uma unica inferencia pode usar ate 8 threads internas
  para executar operadores da rede.

Para comparacao estrita "1 task / 1 thread / batch 1", rode primeiro
o benchmark CPU com intra-op=1 e inter-op=1.

Saida:
  resultados_comparacao/
    configuracao_execucao.csv
    comparacao_metricas.csv
    comparacao_model_only.csv
    comparacao_end_to_end.csv
    comparacao_metricas.png
    comparacao_model_only_fps.png
    comparacao_model_only_latencia.png
    comparacao_end_to_end_fps.png
    comparacao_end_to_end_latencia.png
    resumo_comparacao.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# REFERENCIA DOCUMENTADA - STARCOP_mini
# ============================================================

DOC_TP = 40310
DOC_FP = 4847
DOC_FN = 3467
DOC_TN = 2310672


# ============================================================
# HELPERS
# ============================================================

def metrics_from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = map(int, (tp, fp, fn, tn))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "total_pixels": total,
    }


def find_first(base: Path, filename: str) -> Path | None:
    if not base.exists():
        return None
    hits = sorted(base.rglob(filename))
    return hits[0] if hits else None


def find_required(base: Path, filename: str, label: str) -> Path:
    p = find_first(base, filename)
    if p is None:
        raise FileNotFoundError(
            f"Nao encontrei {filename} em {base} ({label})."
        )
    return p


def pick(row, names, default=np.nan):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                pass
    return default


def load_metrics(path: Path, source: str) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"CSV vazio: {path}")

    # metricas_globais.csv
    row = df.iloc[0]

    tp = pick(row, ["TP", "tp"])
    fp = pick(row, ["FP", "fp"])
    fn = pick(row, ["FN", "fn"])
    tn = pick(row, ["TN", "tn"])

    if all(pd.notna(v) for v in [tp, fp, fn, tn]):
        result = metrics_from_counts(tp, fp, fn, tn)
    else:
        result = {
            "TP": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "TN": np.nan,
            "total_pixels": np.nan,
            "precision": pick(row, ["precision_global", "precision"]),
            "recall": pick(row, ["recall_global", "recall"]),
            "f1": pick(row, ["f1_global", "f1"]),
            "iou": pick(row, ["iou_global", "iou"]),
            "accuracy": pick(row, ["accuracy_global", "accuracy"]),
        }

    result["fonte"] = source
    result["arquivo"] = str(path)
    return result


def load_benchmark(path: Path, platform_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"CSV vazio: {path}")

    rows = []

    for _, r in df.iterrows():
        mode = str(r.get("mode", r.get("modo", "unknown")))

        rows.append({
            "plataforma": platform_name,
            "mode": mode,
            "samples": pick(r, ["samples"]),
            "fps_average": pick(r, ["fps_average", "fps_avg"]),
            "fps_min": pick(r, ["fps_min"]),
            "fps_max": pick(r, ["fps_max"]),
            "fps_p95": pick(r, ["fps_p95"]),
            "fps_p99": pick(r, ["fps_p99"]),
            "latency_avg_ms": pick(r, ["latency_avg_ms"]),
            "latency_min_ms": pick(r, ["latency_min_ms"]),
            "latency_max_ms": pick(r, ["latency_max_ms"]),
            "latency_p95_ms": pick(r, ["latency_p95_ms"]),
            "latency_p99_ms": pick(r, ["latency_p99_ms"]),
        })

    return pd.DataFrame(rows)


def load_cpu_system_info(cpu_root: Path) -> dict:
    p = find_first(cpu_root, "system_info.json")

    if p is None:
        return {
            "file": "",
            "batch_size": 1,
            "torch_num_threads": np.nan,
            "torch_num_interop_threads": np.nan,
            "processor": "desconhecido",
        }

    data = json.loads(p.read_text(encoding="utf-8"))

    return {
        "file": str(p),
        "batch_size": data.get("batch_size", 1),
        "torch_num_threads": data.get("torch_num_threads", np.nan),
        "torch_num_interop_threads": data.get(
            "torch_num_interop_threads", np.nan
        ),
        "processor": data.get("processor", "desconhecido"),
    }


def filter_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    return df[df["mode"].astype(str).str.lower() == mode.lower()].copy()


# ============================================================
# PLOTS
# ============================================================

def bar_plot(df, xcol, value_cols, labels, title, ylabel, out):
    x = np.arange(len(df))
    width = 0.8 / len(value_cols)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (col, label) in enumerate(zip(value_cols, labels)):
        offset = (i - (len(value_cols)-1)/2) * width
        bars = ax.bar(x + offset, df[col], width, label=label)

        for b, v in zip(bars, df[col]):
            if pd.notna(v):
                ax.text(
                    b.get_x() + b.get_width()/2,
                    b.get_height(),
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(df[xcol])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def metrics_plot(df, out):
    metrics = ["precision", "recall", "f1", "iou", "accuracy"]
    labels = ["Precision", "Recall", "F1", "IoU", "Accuracy"]

    x = np.arange(len(metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(
            x + (i - (len(df)-1)/2) * width,
            vals,
            width,
            label=row["fonte"],
        )

        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width()/2,
                b.get_height() + 0.003,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_ylim(0.75, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor")
    ax.set_title("HyperSTARCOP - CPU FP32 x Documentacao x ZCU104 INT8")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )

    args = parser.parse_args()
    root = args.root.resolve()

    cpu_root = root / "resultados_cpu"
    zcu_root = root / "resultados_zcu104"
    out = root / "resultados_comparacao"

    out.mkdir(parents=True, exist_ok=True)

    # Prefere explicitamente o resultado CPU single-thread, se existir.
    cpu_1t_root = cpu_root / "hyperstarcop_cpu_results_1thread"

    if cpu_1t_root.exists():
        cpu_search_root = cpu_1t_root
    else:
        cpu_search_root = cpu_root

    cpu_metrics_file = find_required(
        cpu_search_root,
        "metricas_globais.csv",
        "CPU",
    )

    cpu_bench_file = find_required(
        cpu_search_root,
        "benchmark_summary.csv",
        "CPU",
    )

    zcu_metrics_file = find_required(
        zcu_root,
        "metricas_globais.csv",
        "ZCU104",
    )

    zcu_bench_file = find_required(
        zcu_root,
        "benchmark_summary.csv",
        "ZCU104",
    )

    cpu_info = load_cpu_system_info(cpu_search_root)

    cpu_metrics = load_metrics(
        cpu_metrics_file,
        "CPU FP32",
    )

    doc_metrics = metrics_from_counts(
        DOC_TP,
        DOC_FP,
        DOC_FN,
        DOC_TN,
    )
    doc_metrics["fonte"] = "Documentacao"
    doc_metrics["arquivo"] = "referencia embutida"

    zcu_metrics = load_metrics(
        zcu_metrics_file,
        "ZCU104 INT8",
    )

    metrics_df = pd.DataFrame(
        [cpu_metrics, doc_metrics, zcu_metrics]
    )

    metrics_df.to_csv(
        out / "comparacao_metricas.csv",
        index=False,
    )

    metrics_plot(
        metrics_df,
        out / "comparacao_metricas.png",
    )

    cpu_bench = load_benchmark(
        cpu_bench_file,
        "CPU FP32",
    )

    zcu_bench = load_benchmark(
        zcu_bench_file,
        "ZCU104 INT8",
    )

    # ========================================================
    # CONFIGURACAO DE EXECUCAO
    # ========================================================

    cpu_threads = cpu_info["torch_num_threads"]
    cpu_interop = cpu_info["torch_num_interop_threads"]

    config_rows = [
        {
            "plataforma": "CPU FP32",
            "batch": int(cpu_info.get("batch_size", 1)),
            "inference_tasks_simultaneas": 1,
            "host_threads_do_benchmark": 1,
            "compute_threads_intra_op": cpu_threads,
            "compute_threads_inter_op": cpu_interop,
            "runner_workers": 0,
            "inferences_in_flight": 1,
            "pipeline_tasks": "nao",
            "execucao": (
                "1 inferencia por vez; operadores PyTorch podem "
                f"usar ate {cpu_threads} threads intra-op"
            ),
        },
        {
            "plataforma": "ZCU104 INT8",
            "batch": 1,
            "inference_tasks_simultaneas": 1,
            "host_threads_do_benchmark": 1,
            "compute_threads_intra_op": np.nan,
            "compute_threads_inter_op": np.nan,
            "runner_workers": 1,
            "inferences_in_flight": 1,
            "pipeline_tasks": "nao",
            "execucao": (
                "1 host thread; 1 VART Runner; execute_async seguido "
                "imediatamente de wait; sem overlap"
            ),
        },
    ]

    config_df = pd.DataFrame(config_rows)

    config_df.to_csv(
        out / "configuracao_execucao.csv",
        index=False,
    )

    # ========================================================
    # MODEL ONLY
    # ========================================================

    mo = pd.concat(
        [
            filter_mode(cpu_bench, "model_only"),
            filter_mode(zcu_bench, "model_only"),
        ],
        ignore_index=True,
    )

    mo = mo.merge(
        config_df[[
            "plataforma",
            "batch",
            "inference_tasks_simultaneas",
            "host_threads_do_benchmark",
            "compute_threads_intra_op",
            "runner_workers",
            "inferences_in_flight",
            "pipeline_tasks",
        ]],
        on="plataforma",
        how="left",
    )

    mo.to_csv(
        out / "comparacao_model_only.csv",
        index=False,
    )

    bar_plot(
        mo,
        "plataforma",
        [
            "fps_average",
            "fps_min",
            "fps_max",
            "fps_p95",
            "fps_p99",
        ],
        ["Average", "Min", "Max", "P95", "P99"],
        "Model Only - Batch 1 - 1 inferencia em voo",
        "FPS",
        out / "comparacao_model_only_fps.png",
    )

    bar_plot(
        mo,
        "plataforma",
        [
            "latency_avg_ms",
            "latency_min_ms",
            "latency_max_ms",
            "latency_p95_ms",
            "latency_p99_ms",
        ],
        ["Average", "Min", "Max", "P95", "P99"],
        "Model Only - Latencia - Batch 1",
        "Latencia (ms)",
        out / "comparacao_model_only_latencia.png",
    )

    # ========================================================
    # END TO END
    # ========================================================

    e2e = pd.concat(
        [
            filter_mode(cpu_bench, "end_to_end"),
            filter_mode(zcu_bench, "end_to_end"),
        ],
        ignore_index=True,
    )

    e2e = e2e.merge(
        config_df[[
            "plataforma",
            "batch",
            "inference_tasks_simultaneas",
            "host_threads_do_benchmark",
            "compute_threads_intra_op",
            "runner_workers",
            "inferences_in_flight",
            "pipeline_tasks",
        ]],
        on="plataforma",
        how="left",
    )

    e2e.to_csv(
        out / "comparacao_end_to_end.csv",
        index=False,
    )

    bar_plot(
        e2e,
        "plataforma",
        [
            "fps_average",
            "fps_min",
            "fps_max",
            "fps_p95",
            "fps_p99",
        ],
        ["Average", "Min", "Max", "P95", "P99"],
        "End-to-End - Batch 1 - Pipeline sequencial",
        "FPS",
        out / "comparacao_end_to_end_fps.png",
    )

    bar_plot(
        e2e,
        "plataforma",
        [
            "latency_avg_ms",
            "latency_min_ms",
            "latency_max_ms",
            "latency_p95_ms",
            "latency_p99_ms",
        ],
        ["Average", "Min", "Max", "P95", "P99"],
        "End-to-End - Latencia - Batch 1",
        "Latencia (ms)",
        out / "comparacao_end_to_end_latencia.png",
    )

    # ========================================================
    # SPEEDUPS
    # ========================================================

    def extract_one(df, platform, col):
        r = df[df["plataforma"] == platform]
        if r.empty:
            return np.nan
        return float(r.iloc[0][col])

    speedups = []

    for mode_name, df_mode in [
        ("model_only", mo),
        ("end_to_end", e2e),
    ]:
        cpu_fps = extract_one(df_mode, "CPU FP32", "fps_average")
        zcu_fps = extract_one(df_mode, "ZCU104 INT8", "fps_average")
        cpu_lat = extract_one(df_mode, "CPU FP32", "latency_avg_ms")
        zcu_lat = extract_one(df_mode, "ZCU104 INT8", "latency_avg_ms")

        speedups.append({
            "mode": mode_name,
            "cpu_fps_average": cpu_fps,
            "zcu104_fps_average": zcu_fps,
            "zcu104_vs_cpu_throughput_ratio": (
                zcu_fps / cpu_fps if cpu_fps else np.nan
            ),
            "cpu_latency_avg_ms": cpu_lat,
            "zcu104_latency_avg_ms": zcu_lat,
            "cpu_vs_zcu104_latency_ratio": (
                cpu_lat / zcu_lat if zcu_lat else np.nan
            ),
        })

    speedup_df = pd.DataFrame(speedups)
    speedup_df.to_csv(
        out / "comparacao_speedup.csv",
        index=False,
    )

    # ========================================================
    # RESUMO
    # ========================================================

    strict_single_thread = (
        cpu_threads == 1
        and (
            pd.isna(cpu_interop)
            or cpu_interop == 1
        )
    )

    lines = []
    lines.append("HYPERSTARCOP - COMPARACAO CPU x DOCUMENTACAO x ZCU104")
    lines.append("=" * 76)
    lines.append("")

    lines.append("CONFIGURACAO DE EXECUCAO")
    lines.append(f"Resultado CPU selecionado: {cpu_search_root}")
    lines.append("-" * 76)
    lines.append(
        "CPU FP32:"
    )
    lines.append(
        f"  batch = {cpu_info.get('batch_size', 1)}"
    )
    lines.append(
        "  inference tasks simultaneas = 1"
    )
    lines.append(
        "  host benchmark thread = 1"
    )
    lines.append(
        f"  PyTorch intra-op threads = {cpu_threads}"
    )
    lines.append(
        f"  PyTorch inter-op threads = {cpu_interop}"
    )
    lines.append(
        "  inferences in flight = 1"
    )
    lines.append("")

    lines.append(
        "ZCU104 INT8:"
    )
    lines.append(
        "  batch = 1"
    )
    lines.append(
        "  host threads = 1"
    )
    lines.append(
        "  VART runners = 1"
    )
    lines.append(
        "  inference tasks simultaneas = 1"
    )
    lines.append(
        "  inferences in flight = 1"
    )
    lines.append(
        "  pipeline/tasks paralelas = nao"
    )
    lines.append(
        "  execute_async() seguido imediatamente de wait()"
    )
    lines.append("")

    if strict_single_thread:
        lines.append(
            "STATUS: CPU configurada em single-thread para comparacao estrita."
        )
    else:
        lines.append(
            "ATENCAO: CPU NAO esta em single-thread estrito."
        )
        lines.append(
            f"Ela usa {cpu_threads} threads PyTorch intra-op para UMA inferencia."
        )
        lines.append(
            "Isso continua sendo batch=1 e uma task por vez, mas nao 1 compute thread."
        )
        lines.append(
            "Para comparacao 1-thread, refaca o benchmark CPU com intra-op=1 e inter-op=1."
        )

    lines.append("")
    lines.append("METRICAS")
    lines.append("-" * 76)
    lines.append(
        metrics_df[
            ["fonte", "precision", "recall", "f1", "iou", "accuracy"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    lines.append("")
    lines.append("MODEL ONLY")
    lines.append("-" * 76)
    lines.append(
        mo[
            [
                "plataforma",
                "batch",
                "inference_tasks_simultaneas",
                "compute_threads_intra_op",
                "fps_average",
                "latency_avg_ms",
                "latency_p95_ms",
                "latency_p99_ms",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    lines.append("")
    lines.append("END TO END")
    lines.append("-" * 76)
    lines.append(
        e2e[
            [
                "plataforma",
                "batch",
                "inference_tasks_simultaneas",
                "compute_threads_intra_op",
                "fps_average",
                "latency_avg_ms",
                "latency_p95_ms",
                "latency_p99_ms",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    lines.append("")
    lines.append("SPEEDUP / RAZOES")
    lines.append("-" * 76)
    lines.append(
        speedup_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    summary = "\n".join(lines)

    (out / "resumo_comparacao.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)

    print()
    print("Arquivos gerados em:")
    print(out)


if __name__ == "__main__":
    main()
