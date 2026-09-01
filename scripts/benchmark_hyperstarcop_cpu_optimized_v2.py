#!/usr/bin/env python3
"""
HyperSTARCOP - CPU FP32 optimized benchmark
Batch is ALWAYS 1.

Profiles:
  baseline
      - sequential
      - one inference in flight
      - reports:
          baseline_model_only
          baseline_end_to_end

  max-model-only
      - inputs preloaded/preprocessed before timing
      - multiple concurrent inference workers
      - one model copy per worker
      - batch=1 for every inference
      - throughput = completed / wall time

  max-e2e
      - threaded pipeline:
          PRE workers -> INFERENCE workers -> POST workers
      - TIFF disk I/O INCLUDED
      - normalization INCLUDED
      - model forward INCLUDED
      - sigmoid + threshold INCLUDED
      - labels/metrics/CSV are OUTSIDE timed pipeline

Validation:
  TP, FP, FN, TN, Precision, Recall, F1, IoU, Accuracy

Statistics:
  throughput FPS (wall)
  latency mean / median / min / max / p90 / p95 / p99
  stddev / CV / p99-p50 jitter
  inter-completion interval mean/p95/p99
  stage timing means/p95/p99

Examples:
  python3 benchmark_hyperstarcop_cpu_optimized.py --profile all \
      --workers 8 --pre-workers 2 --post-workers 2 \
      --intra-threads 1 --interop-threads 1 --iterations 500 --pin

Aliases:
  --baseline
  --maxthroughputmodelonly
  --maxendend
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import queue
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Defaults avoid nested BLAS/OpenMP oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

try:
    import rasterio
except Exception:
    rasterio = None

try:
    import tifffile
except Exception:
    tifffile = None

try:
    import cv2
except Exception:
    cv2 = None


# ============================================================================
# CONSTANTS
# ============================================================================

BATCH = 1
H = 512
W = 512
C = 4

DOC_TP = 40310
DOC_FP = 4847
DOC_FN = 3467
DOC_TN = 2310672


# ============================================================================
# OPTIONS
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="HyperSTARCOP CPU FP32 optimized benchmark, batch=1 always"
    )

    p.add_argument(
        "--profile",
        choices=["all", "baseline", "max-model-only", "max-e2e"],
        default="all",
    )

    # Aliases requested by user.
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--maxthroughputmodelonly", action="store_true")
    p.add_argument("--maxendend", action="store_true")

    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)

    # Concurrency.
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pre-workers", type=int, default=2)
    p.add_argument("--post-workers", type=int, default=1)
    p.add_argument("--slots", type=int, default=8)

    # Torch internal parallelism.
    p.add_argument("--intra-threads", type=int, default=1)
    p.add_argument("--interop-threads", type=int, default=1)

    # Benchmark.
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--worker-warmup",
        type=int,
        default=5,
        help=(
            "Warmup por worker nos perfis concorrentes. "
            "Executado em paralelo. Padrao: 5."
        ),
    )
    p.add_argument("--baseline-repeats", type=int, default=100)
    p.add_argument("--baseline-e2e-passes", type=int, default=5)

    p.add_argument("--pin", action="store_true")
    p.add_argument("--no-pin", dest="pin", action="store_false")
    p.set_defaults(pin=False)

    p.add_argument("--validate", action="store_true")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.set_defaults(validate=True)

    args = p.parse_args()

    if args.baseline:
        args.profile = "baseline"
    if args.maxthroughputmodelonly:
        args.profile = "max-model-only"
    if args.maxendend:
        args.profile = "max-e2e"

    if min(
        args.workers,
        args.pre_workers,
        args.post_workers,
        args.slots,
        args.iterations,
        args.baseline_repeats,
        args.baseline_e2e_passes,
        args.intra_threads,
        args.interop_threads,
    ) <= 0:
        raise ValueError("All numeric concurrency/benchmark values must be > 0")

    if args.warmup < 0 or args.worker_warmup < 0:
        raise ValueError("--warmup and --worker-warmup must be >= 0")

    root = args.root.resolve()

    args.dataset = (
        args.dataset.resolve()
        if args.dataset is not None
        else root / "STARCOP_mini"
    )

    args.csv = (
        args.csv.resolve()
        if args.csv is not None
        else args.dataset / "test_mini10.csv"
    )

    args.weights = (
        args.weights.resolve()
        if args.weights is not None
        else (
            root
            / "vitis_ai"
            / "float_model"
            / "hyperstarcop_network_fp32.pth"
        )
    )

    args.out = (
        args.out.resolve()
        if args.out is not None
        else root / "resultados_cpu" / "hyperstarcop_cpu_optimized"
    )

    return args


# ============================================================================
# SYSTEM / AFFINITY
# ============================================================================

def configure_torch(args):
    torch.set_num_threads(args.intra_threads)

    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        # PyTorch only permits this before parallel work starts.
        pass


def pin_current_thread(slot: int):
    """
    Best-effort Linux thread affinity.

    Linux sched_setaffinity accepts a TID. threading.get_native_id()
    returns the current Linux thread id.
    """
    try:
        ncpu = os.cpu_count() or 1
        core = slot % ncpu
        tid = threading.get_native_id()
        os.sched_setaffinity(tid, {core})
    except Exception as exc:
        print(f"WARN affinity failed for slot={slot}: {exc}")


def save_system_info(args, model):
    info = {
        "platform": "CPU",
        "precision": "FP32",
        "batch": BATCH,
        "python": platform.python_version(),
        "platform_string": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "os_cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "profile": args.profile,
        "workers": args.workers,
        "pre_workers": args.pre_workers,
        "post_workers": args.post_workers,
        "slots": args.slots,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "worker_warmup": args.worker_warmup,
        "pin": args.pin,
        "weights": str(args.weights),
        "dataset": str(args.dataset),
        "csv": str(args.csv),
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    args.out.mkdir(parents=True, exist_ok=True)

    (args.out / "config.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )


# ============================================================================
# MODEL
# ============================================================================

def create_model(weights: Path):
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=4,
        classes=1,
        activation=None,
    )

    state = torch.load(
        weights,
        map_location="cpu",
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()
    model.to("cpu")

    return model


# ============================================================================
# DATA IO / PREPROCESS
# ============================================================================

def read_tif(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    if rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32, copy=False)

    elif tifffile is not None:
        arr = tifffile.imread(path).astype(np.float32, copy=False)

    elif cv2 is not None:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if arr is None:
            raise RuntimeError(f"OpenCV could not read {path}")

        if arr.ndim != 2:
            raise RuntimeError(f"Expected one-channel TIFF: {path}")

        arr = arr.astype(np.float32, copy=False)

    else:
        raise RuntimeError(
            "Need at least one TIFF reader: rasterio, tifffile or cv2"
        )

    if arr.shape != (H, W):
        raise RuntimeError(
            f"Expected {(H, W)}, got {arr.shape}: {path}"
        )

    return np.ascontiguousarray(arr, dtype=np.float32)


def preprocess(folder: Path) -> torch.Tensor:
    mag1c = read_tif(folder / "mag1c.tif")
    red = read_tif(folder / "TOA_AVIRIS_640nm.tif")
    green = read_tif(folder / "TOA_AVIRIS_550nm.tif")
    blue = read_tif(folder / "TOA_AVIRIS_460nm.tif")

    # Standalone HyperSTARCOP / Vitis preprocessing.
    mag1c = np.clip(mag1c / 1750.0, 0.0, 2.0)
    red = np.clip(red / 60.0, 0.0, 2.0)
    green = np.clip(green / 60.0, 0.0, 2.0)
    blue = np.clip(blue / 60.0, 0.0, 2.0)

    x = np.stack(
        [mag1c, red, green, blue],
        axis=0,
    )

    x = np.ascontiguousarray(
        x,
        dtype=np.float32,
    )

    x = torch.from_numpy(x).unsqueeze(0)

    if tuple(x.shape) != (1, 4, H, W):
        raise RuntimeError(
            f"Batch/shape error: got {tuple(x.shape)}, "
            f"expected {(1, 4, H, W)}"
        )

    return x


def read_label(folder: Path) -> np.ndarray:
    x = read_tif(folder / "labelbinary.tif")
    return (x > 0).astype(np.uint8)


# ============================================================================
# METRICS
# ============================================================================

def metrics_from_counts(tp, fp, fn, tn):
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn
        else 0.0
    )

    total = tp + fp + fn + tn

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


def confusion(pred, label):
    tp = int(np.sum((pred == 1) & (label == 1)))
    fp = int(np.sum((pred == 1) & (label == 0)))
    fn = int(np.sum((pred == 0) & (label == 1)))
    tn = int(np.sum((pred == 0) & (label == 0)))
    return tp, fp, fn, tn


# ============================================================================
# STATISTICS
# ============================================================================

@dataclass
class Stats:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    stddev: float = 0.0
    cv: float = 0.0


def summarize(values) -> Stats:
    arr = np.asarray(
        list(values),
        dtype=np.float64,
    )

    if arr.size == 0:
        return Stats()

    mean = float(arr.mean())
    stddev = float(arr.std(ddof=0))

    return Stats(
        count=int(arr.size),
        mean=mean,
        median=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        min=float(arr.min()),
        max=float(arr.max()),
        stddev=stddev,
        cv=(stddev / mean if mean != 0.0 else 0.0),
    )


@dataclass
class Sample:
    job: int
    dataset_index: int

    worker: int = -1
    pre_worker: int = -1
    post_worker: int = -1

    slot_wait_ms: float = 0.0
    io_preprocess_ms: float = 0.0
    pre_infer_queue_ms: float = 0.0
    model_ms: float = 0.0
    infer_post_queue_ms: float = 0.0
    postprocess_ms: float = 0.0
    e2e_ms: float = 0.0
    completion_s: float = 0.0


@dataclass
class BenchmarkResult:
    mode: str
    batch: int
    workers: int
    pre_workers: int
    post_workers: int
    slots: int

    completed: int
    wall_s: float
    throughput_fps: float

    latency: Stats
    model: Stats
    preprocess: Stats
    postprocess: Stats
    inter_completion: Stats

    samples: list[Sample]


def finalize_result(
    *,
    mode,
    workers,
    pre_workers,
    post_workers,
    slots,
    completed,
    wall_s,
    samples,
):
    lat = [s.e2e_ms for s in samples]
    model = [s.model_ms for s in samples if s.model_ms > 0]
    prep = [s.io_preprocess_ms for s in samples if s.io_preprocess_ms > 0]
    post = [s.postprocess_ms for s in samples if s.postprocess_ms > 0]

    completion = sorted(
        s.completion_s
        for s in samples
    )

    intervals = []

    for i in range(1, len(completion)):
        intervals.append(
            (completion[i] - completion[i - 1])
            * 1000.0
        )

    return BenchmarkResult(
        mode=mode,
        batch=BATCH,
        workers=workers,
        pre_workers=pre_workers,
        post_workers=post_workers,
        slots=slots,
        completed=completed,
        wall_s=wall_s,
        throughput_fps=completed / wall_s,
        latency=summarize(lat),
        model=summarize(model),
        preprocess=summarize(prep),
        postprocess=summarize(post),
        inter_completion=summarize(intervals),
        samples=samples,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def run_validation(args, dataset_df, model):
    rows = []

    TP = FP = FN = TN = 0

    with torch.inference_mode():
        for i, row in dataset_df.iterrows():
            sample_id = row["id"]
            folder = args.dataset / sample_id

            x = preprocess(folder)

            logits = model(x)

            prob = torch.sigmoid(logits)

            pred = (
                prob > 0.5
            ).to(torch.uint8)

            pred_np = (
                pred[0, 0]
                .cpu()
                .numpy()
            )

            label = read_label(folder)

            tp, fp, fn, tn = confusion(
                pred_np,
                label,
            )

            TP += tp
            FP += fp
            FN += fn
            TN += tn

            m = metrics_from_counts(
                tp,
                fp,
                fn,
                tn,
            )

            rows.append(
                {
                    "id": sample_id,
                    "has_plume": row.get("has_plume", ""),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    **m,
                }
            )

            print(
                f"[VAL {i+1}/{len(dataset_df)}] "
                f"{sample_id} "
                f"F1={m['f1']:.4f} "
                f"IoU={m['iou']:.4f}"
            )

    gm = metrics_from_counts(
        TP,
        FP,
        FN,
        TN,
    )

    pd.DataFrame(rows).to_csv(
        args.out / "metricas_por_imagem.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "num_imagens": len(dataset_df),
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "precision_global": gm["precision"],
                "recall_global": gm["recall"],
                "f1_global": gm["f1"],
                "iou_global": gm["iou"],
                "accuracy_global": gm["accuracy"],
            }
        ]
    ).to_csv(
        args.out / "metricas_globais.csv",
        index=False,
    )

    doc = metrics_from_counts(
        DOC_TP,
        DOC_FP,
        DOC_FN,
        DOC_TN,
    )

    comparison = []

    for name in [
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
    ]:
        d = doc[name]
        c = gm[name]

        comparison.append(
            {
                "metrica": name,
                "documentacao": d,
                "cpu_fp32": c,
                "delta_abs": c - d,
                "delta_pct": (
                    (c - d) / d * 100.0
                    if d != 0
                    else np.nan
                ),
            }
        )

    pd.DataFrame(comparison).to_csv(
        args.out / "comparacao_documentacao.csv",
        index=False,
    )

    print()
    print("CPU FP32 VALIDATION GLOBAL")
    print(
        f"Precision={gm['precision']:.6f} "
        f"Recall={gm['recall']:.6f} "
        f"F1={gm['f1']:.6f} "
        f"IoU={gm['iou']:.6f} "
        f"Accuracy={gm['accuracy']:.6f}"
    )


# ============================================================================
# PREPARED INPUT CACHE
# ============================================================================

def prepare_inputs(args, dataset_df):
    prepared = []

    for _, row in dataset_df.iterrows():
        prepared.append(
            preprocess(
                args.dataset / row["id"]
            )
        )

    return prepared


# ============================================================================
# BASELINE
# ============================================================================

def run_baseline(args, dataset_df, base_model):
    print("[BASELINE] preparing normalized FP32 inputs...")
    prepared = prepare_inputs(
        args,
        dataset_df,
    )

    print(
        f"[BASELINE MODEL-ONLY] warmup={args.warmup}, "
        f"measured={args.baseline_repeats}"
    )

    # ------------------------------------------------------------------------
    # Model-only sequential.
    # ------------------------------------------------------------------------

    with torch.inference_mode():
        for i in range(args.warmup):
            x = prepared[
                i % len(prepared)
            ]
            _ = base_model(x)

        samples = []

        wall0 = time.perf_counter()

        for i in range(args.baseline_repeats):
            x = prepared[
                i % len(prepared)
            ]

            t0 = time.perf_counter()

            _ = base_model(x)

            t1 = time.perf_counter()

            ms = (t1 - t0) * 1000.0

            samples.append(
                Sample(
                    job=i,
                    dataset_index=i % len(prepared),
                    worker=0,
                    model_ms=ms,
                    e2e_ms=ms,
                    completion_s=t1 - wall0,
                )
            )

        wall1 = time.perf_counter()

    print("[BASELINE MODEL-ONLY] measurements finished")

    mo = finalize_result(
        mode="baseline_model_only",
        workers=1,
        pre_workers=0,
        post_workers=0,
        slots=1,
        completed=len(samples),
        wall_s=wall1 - wall0,
        samples=samples,
    )

    # ------------------------------------------------------------------------
    # E2E sequential.
    # ------------------------------------------------------------------------

    print(
        f"[BASELINE E2E] warmup=1 dataset pass, "
        f"measured={args.baseline_e2e_passes} passes "
        f"({args.baseline_e2e_passes * len(dataset_df)} images)"
    )

    # Warm one full pass.
    with torch.inference_mode():
        for _, row in dataset_df.iterrows():
            x = preprocess(
                args.dataset / row["id"]
            )

            logits = base_model(x)

            pred = (
                torch.sigmoid(logits) > 0.5
            ).to(torch.uint8)

            _ = int(
                pred[0, 0, 0, 0].item()
            )

    samples = []

    wall0 = time.perf_counter()

    job = 0

    with torch.inference_mode():
        for _pass in range(args.baseline_e2e_passes):
            for di, row in dataset_df.iterrows():
                folder = args.dataset / row["id"]

                e0 = time.perf_counter()

                p0 = time.perf_counter()
                x = preprocess(folder)
                p1 = time.perf_counter()

                m0 = time.perf_counter()
                logits = base_model(x)
                m1 = time.perf_counter()

                q0 = time.perf_counter()

                pred = (
                    torch.sigmoid(logits) > 0.5
                ).to(torch.uint8)

                _ = int(
                    pred[0, 0, 0, 0].item()
                )

                q1 = time.perf_counter()

                samples.append(
                    Sample(
                        job=job,
                        dataset_index=int(di),
                        worker=0,
                        pre_worker=0,
                        post_worker=0,
                        io_preprocess_ms=(p1 - p0) * 1000.0,
                        model_ms=(m1 - m0) * 1000.0,
                        postprocess_ms=(q1 - q0) * 1000.0,
                        e2e_ms=(q1 - e0) * 1000.0,
                        completion_s=q1 - wall0,
                    )
                )

                job += 1

    wall1 = time.perf_counter()

    e2e = finalize_result(
        mode="baseline_end_to_end",
        workers=1,
        pre_workers=1,
        post_workers=1,
        slots=1,
        completed=len(samples),
        wall_s=wall1 - wall0,
        samples=samples,
    )

    return [mo, e2e]


# ============================================================================
# PARALLEL WORKER WARMUP
# ============================================================================

def warmup_models_parallel(args, models, prepared, label):
    """
    Warm each independent model copy concurrently.

    This avoids the previous behavior:
        workers * warmup
    being executed serially and appearing to hang.
    """
    if args.worker_warmup <= 0:
        return

    n = len(models)

    print()
    print(
        f"[{label}] parallel warmup: "
        f"{n} workers x {args.worker_warmup} iterations"
    )

    barrier = threading.Barrier(n + 1)
    done = [0 for _ in range(n)]

    def fn(worker_id):
        if args.pin:
            pin_current_thread(worker_id)

        model = models[worker_id]
        x = prepared[worker_id % len(prepared)]

        barrier.wait()

        with torch.inference_mode():
            for i in range(args.worker_warmup):
                _ = model(x)
                done[worker_id] = i + 1

        print(
            f"  warmup worker {worker_id}: "
            f"{args.worker_warmup}/{args.worker_warmup} done"
        )

    threads = [
        threading.Thread(target=fn, args=(i,))
        for i in range(n)
    ]

    for t in threads:
        t.start()

    t0 = time.perf_counter()
    barrier.wait()

    for t in threads:
        t.join()

    print(
        f"[{label}] warmup finished in "
        f"{time.perf_counter() - t0:.2f} s"
    )


# ============================================================================
# MAX MODEL-ONLY THROUGHPUT
# ============================================================================

def run_max_model_only(args, dataset_df, base_model):
    prepared = prepare_inputs(
        args,
        dataset_df,
    )

    # Independent model copy per inference worker.
    models = [
        copy.deepcopy(base_model).eval()
        for _ in range(args.workers)
    ]

    warmup_models_parallel(
        args,
        models,
        prepared,
        "MAX MODEL-ONLY",
    )

    print(
        f"[MAX MODEL-ONLY] measured run: "
        f"{args.iterations} total inferences, "
        f"{args.workers} concurrent workers"
    )

    samples: list[Optional[Sample]] = [
        None
        for _ in range(args.iterations)
    ]

    next_lock = threading.Lock()
    next_job = 0

    start_event = threading.Event()
    ready = threading.Barrier(args.workers + 1)

    run_start_holder = {"t": 0.0}

    def worker_fn(worker_id: int):
        nonlocal next_job

        if args.pin:
            pin_current_thread(worker_id)

        model = models[worker_id]

        ready.wait()
        start_event.wait()

        with torch.inference_mode():
            while True:
                with next_lock:
                    j = next_job
                    next_job += 1

                if j >= args.iterations:
                    return

                x = prepared[
                    j % len(prepared)
                ]

                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()

                ms = (t1 - t0) * 1000.0

                samples[j] = Sample(
                    job=j,
                    dataset_index=j % len(prepared),
                    worker=worker_id,
                    model_ms=ms,
                    e2e_ms=ms,
                    completion_s=(
                        t1 - run_start_holder["t"]
                    ),
                )

    threads = [
        threading.Thread(
            target=worker_fn,
            args=(i,),
            daemon=False,
        )
        for i in range(args.workers)
    ]

    for t in threads:
        t.start()

    ready.wait()

    run_start = time.perf_counter()
    run_start_holder["t"] = run_start
    start_event.set()

    for t in threads:
        t.join()

    run_end = time.perf_counter()

    concrete = [
        s for s in samples
        if s is not None
    ]

    return finalize_result(
        mode="max_model_only_throughput",
        workers=args.workers,
        pre_workers=0,
        post_workers=0,
        slots=args.workers,
        completed=len(concrete),
        wall_s=run_end - run_start,
        samples=concrete,
    )


# ============================================================================
# MAX E2E PIPELINE
# ============================================================================

@dataclass
class Frame:
    job: int
    dataset_index: int
    arrival: float

    x: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    pre_worker: int = -1
    worker: int = -1
    post_worker: int = -1

    queued_infer: float = 0.0
    queued_post: float = 0.0

    io_preprocess_ms: float = 0.0
    pre_infer_queue_ms: float = 0.0
    model_ms: float = 0.0
    infer_post_queue_ms: float = 0.0
    postprocess_ms: float = 0.0


def run_max_e2e(args, dataset_df, base_model):
    models = [
        copy.deepcopy(base_model).eval()
        for _ in range(args.workers)
    ]

    # Warm model workers concurrently.
    warm_x = preprocess(
        args.dataset
        / dataset_df.iloc[0]["id"]
    )

    warmup_models_parallel(
        args,
        models,
        [warm_x],
        "MAX END-TO-END",
    )

    print(
        f"[MAX END-TO-END] measured run: "
        f"{args.iterations} images | "
        f"pre={args.pre_workers} infer={args.workers} "
        f"post={args.post_workers} slots={args.slots}"
    )

    job_q = queue.Queue()
    infer_q = queue.Queue(
        maxsize=args.slots
    )
    post_q = queue.Queue(
        maxsize=args.slots
    )

    for j in range(args.iterations):
        job_q.put(j)

    for _ in range(args.pre_workers):
        job_q.put(None)

    samples: list[Optional[Sample]] = [
        None
        for _ in range(args.iterations)
    ]

    start_event = threading.Event()

    participants = (
        args.pre_workers
        + args.workers
        + args.post_workers
    )

    ready = threading.Barrier(
        participants + 1
    )

    run_start_holder = {"t": 0.0}

    # ------------------------------------------------------------------------
    # PRE
    # ------------------------------------------------------------------------

    def pre_fn(pre_id: int):
        if args.pin:
            pin_current_thread(pre_id)

        ready.wait()
        start_event.wait()

        while True:
            j = job_q.get()

            if j is None:
                return

            di = j % len(dataset_df)

            arrival = time.perf_counter()

            row = dataset_df.iloc[di]

            p0 = time.perf_counter()

            x = preprocess(
                args.dataset
                / row["id"]
            )

            p1 = time.perf_counter()

            frame = Frame(
                job=j,
                dataset_index=di,
                arrival=arrival,
                x=x,
                pre_worker=pre_id,
                queued_infer=time.perf_counter(),
                io_preprocess_ms=(p1 - p0) * 1000.0,
            )

            infer_q.put(frame)

    # ------------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------------

    def infer_fn(worker_id: int):
        if args.pin:
            pin_current_thread(
                args.pre_workers
                + args.post_workers
                + worker_id
            )

        model = models[worker_id]

        ready.wait()
        start_event.wait()

        with torch.inference_mode():
            while True:
                frame = infer_q.get()

                if frame is None:
                    return

                now = time.perf_counter()

                frame.worker = worker_id
                frame.pre_infer_queue_ms = (
                    now - frame.queued_infer
                ) * 1000.0

                m0 = time.perf_counter()

                frame.logits = model(
                    frame.x
                )

                m1 = time.perf_counter()

                frame.model_ms = (
                    m1 - m0
                ) * 1000.0

                frame.x = None
                frame.queued_post = time.perf_counter()

                post_q.put(frame)

    # ------------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------------

    def post_fn(post_id: int):
        if args.pin:
            pin_current_thread(
                args.pre_workers
                + post_id
            )

        ready.wait()
        start_event.wait()

        while True:
            frame = post_q.get()

            if frame is None:
                return

            now = time.perf_counter()

            frame.post_worker = post_id
            frame.infer_post_queue_ms = (
                now - frame.queued_post
            ) * 1000.0

            q0 = time.perf_counter()

            pred = (
                torch.sigmoid(
                    frame.logits
                )
                > 0.5
            ).to(torch.uint8)

            # Materialize final batch=1 mask result.
            _ = int(
                pred[0, 0, 0, 0].item()
            )

            q1 = time.perf_counter()

            frame.postprocess_ms = (
                q1 - q0
            ) * 1000.0

            frame.logits = None

            samples[frame.job] = Sample(
                job=frame.job,
                dataset_index=frame.dataset_index,
                worker=frame.worker,
                pre_worker=frame.pre_worker,
                post_worker=frame.post_worker,
                io_preprocess_ms=frame.io_preprocess_ms,
                pre_infer_queue_ms=frame.pre_infer_queue_ms,
                model_ms=frame.model_ms,
                infer_post_queue_ms=frame.infer_post_queue_ms,
                postprocess_ms=frame.postprocess_ms,
                e2e_ms=(
                    q1 - frame.arrival
                ) * 1000.0,
                completion_s=(
                    q1
                    - run_start_holder["t"]
                ),
            )

    pre_threads = [
        threading.Thread(
            target=pre_fn,
            args=(i,),
        )
        for i in range(args.pre_workers)
    ]

    infer_threads = [
        threading.Thread(
            target=infer_fn,
            args=(i,),
        )
        for i in range(args.workers)
    ]

    post_threads = [
        threading.Thread(
            target=post_fn,
            args=(i,),
        )
        for i in range(args.post_workers)
    ]

    all_threads = (
        pre_threads
        + infer_threads
        + post_threads
    )

    for t in all_threads:
        t.start()

    ready.wait()

    run_start = time.perf_counter()
    run_start_holder["t"] = run_start
    start_event.set()

    # Wait preprocess producers.
    for t in pre_threads:
        t.join()

    # Stop inference workers after all frames are queued.
    for _ in range(args.workers):
        infer_q.put(None)

    for t in infer_threads:
        t.join()

    # Stop post workers after all inference outputs are queued.
    for _ in range(args.post_workers):
        post_q.put(None)

    for t in post_threads:
        t.join()

    run_end = time.perf_counter()

    concrete = [
        s
        for s in samples
        if s is not None
    ]

    if len(concrete) != args.iterations:
        raise RuntimeError(
            f"Pipeline completed {len(concrete)} / {args.iterations} jobs"
        )

    return finalize_result(
        mode="max_end_to_end_throughput",
        workers=args.workers,
        pre_workers=args.pre_workers,
        post_workers=args.post_workers,
        slots=args.slots,
        completed=len(concrete),
        wall_s=run_end - run_start,
        samples=concrete,
    )


# ============================================================================
# OUTPUT
# ============================================================================

def result_to_summary_row(r: BenchmarkResult):
    return {
        "mode": r.mode,
        "batch": r.batch,
        "workers": r.workers,
        "pre_workers": r.pre_workers,
        "post_workers": r.post_workers,
        "slots": r.slots,
        "completed": r.completed,
        "wall_s": r.wall_s,
        "throughput_fps": r.throughput_fps,

        # Per-request FPS equivalent derived from latency.
        # In concurrent profiles this is NOT the sustained system throughput.
        "equiv_fps_avg": (
            1000.0 / r.latency.mean
            if r.latency.mean > 0 else 0.0
        ),
        "equiv_fps_min": (
            1000.0 / r.latency.max
            if r.latency.max > 0 else 0.0
        ),
        "equiv_fps_max": (
            1000.0 / r.latency.min
            if r.latency.min > 0 else 0.0
        ),
        "equiv_fps_p95": (
            1000.0 / r.latency.p95
            if r.latency.p95 > 0 else 0.0
        ),
        "equiv_fps_p99": (
            1000.0 / r.latency.p99
            if r.latency.p99 > 0 else 0.0
        ),

        # Output cadence statistics. Wall throughput is the primary sustained FPS.
        "completion_fps_avg": r.throughput_fps,
        "completion_fps_min": (
            1000.0 / r.inter_completion.max
            if r.inter_completion.max > 0 else r.throughput_fps
        ),
        "completion_fps_max": (
            1000.0 / r.inter_completion.min
            if r.inter_completion.min > 0 else r.throughput_fps
        ),
        "completion_fps_p95": (
            1000.0 / r.inter_completion.p95
            if r.inter_completion.p95 > 0 else r.throughput_fps
        ),
        "completion_fps_p99": (
            1000.0 / r.inter_completion.p99
            if r.inter_completion.p99 > 0 else r.throughput_fps
        ),

        "latency_mean_ms": r.latency.mean,
        "latency_median_ms": r.latency.median,
        "latency_min_ms": r.latency.min,
        "latency_max_ms": r.latency.max,
        "latency_p90_ms": r.latency.p90,
        "latency_p95_ms": r.latency.p95,
        "latency_p99_ms": r.latency.p99,
        "latency_stddev_ms": r.latency.stddev,
        "latency_cv": r.latency.cv,
        "latency_p99_minus_p50_ms": (
            r.latency.p99
            - r.latency.median
        ),

        "model_mean_ms": r.model.mean,
        "model_p95_ms": r.model.p95,
        "model_p99_ms": r.model.p99,

        "preprocess_mean_ms": r.preprocess.mean,
        "preprocess_p95_ms": r.preprocess.p95,
        "preprocess_p99_ms": r.preprocess.p99,

        "postprocess_mean_ms": r.postprocess.mean,
        "postprocess_p95_ms": r.postprocess.p95,
        "postprocess_p99_ms": r.postprocess.p99,

        "inter_completion_mean_ms": r.inter_completion.mean,
        "inter_completion_p95_ms": r.inter_completion.p95,
        "inter_completion_p99_ms": r.inter_completion.p99,
    }


def save_results(args, results):
    summary = pd.DataFrame(
        result_to_summary_row(r)
        for r in results
    )

    summary.to_csv(
        args.out / "benchmark_summary.csv",
        index=False,
    )

    raw_rows = []

    for r in results:
        for s in r.samples:
            row = asdict(s)
            row["mode"] = r.mode
            row["instantaneous_fps"] = (
                1000.0 / s.e2e_ms
                if s.e2e_ms > 0
                else 0.0
            )
            raw_rows.append(row)

    pd.DataFrame(raw_rows).to_csv(
        args.out / "benchmark_samples.csv",
        index=False,
    )


def print_result(r: BenchmarkResult):
    print()
    print("=" * 72)
    print(r.mode)
    print("=" * 72)

    print(f"batch              = {r.batch}")
    print(f"workers            = {r.workers}")
    print(f"pre_workers        = {r.pre_workers}")
    print(f"post_workers       = {r.post_workers}")
    print(f"slots              = {r.slots}")
    print(f"completed          = {r.completed}")
    print(f"wall_s             = {r.wall_s:.6f}")
    print(f"THROUGHPUT FPS     = {r.throughput_fps:.6f}")

    print()
    print("FPS EQUIVALENT FROM PER-JOB LATENCY")
    print(
        f" average = {1000.0 / r.latency.mean:.6f}"
        if r.latency.mean > 0 else " average = 0"
    )
    print(
        f" min     = {1000.0 / r.latency.max:.6f}"
        if r.latency.max > 0 else " min     = 0"
    )
    print(
        f" max     = {1000.0 / r.latency.min:.6f}"
        if r.latency.min > 0 else " max     = 0"
    )
    print(
        f" p95     = {1000.0 / r.latency.p95:.6f}"
        if r.latency.p95 > 0 else " p95     = 0"
    )
    print(
        f" p99     = {1000.0 / r.latency.p99:.6f}"
        if r.latency.p99 > 0 else " p99     = 0"
    )

    print()
    print("LATENCY ms")
    print(f" mean   = {r.latency.mean:.6f}")
    print(f" median = {r.latency.median:.6f}")
    print(f" min    = {r.latency.min:.6f}")
    print(f" max    = {r.latency.max:.6f}")
    print(f" p90    = {r.latency.p90:.6f}")
    print(f" p95    = {r.latency.p95:.6f}")
    print(f" p99    = {r.latency.p99:.6f}")
    print(f" stddev = {r.latency.stddev:.6f}")
    print(f" CV     = {r.latency.cv:.6f}")
    print(
        " jitter p99-p50 = "
        f"{r.latency.p99 - r.latency.median:.6f} ms"
    )

    if r.model.count:
        print()
        print("MODEL STAGE ms")
        print(f" mean = {r.model.mean:.6f}")
        print(f" p95  = {r.model.p95:.6f}")
        print(f" p99  = {r.model.p99:.6f}")

    if r.preprocess.count:
        print()
        print(
            f"PREPROCESS mean = {r.preprocess.mean:.6f} ms"
        )

    if r.postprocess.count:
        print(
            f"POSTPROCESS mean = {r.postprocess.mean:.6f} ms"
        )

    if r.inter_completion.count:
        print()
        print("INTER-COMPLETION ms")
        print(
            f" mean = {r.inter_completion.mean:.6f}"
        )
        print(
            f" p95  = {r.inter_completion.p95:.6f}"
        )
        print(
            f" p99  = {r.inter_completion.p99:.6f}"
        )

        print("OUTPUT-CADENCE FPS")
        print(f" average(system wall) = {r.throughput_fps:.6f}")
        print(
            f" min = {1000.0 / r.inter_completion.max:.6f}"
            if r.inter_completion.max > 0 else " min = 0"
        )
        print(
            f" max = {1000.0 / r.inter_completion.min:.6f}"
            if r.inter_completion.min > 0 else " max = 0"
        )
        print(
            f" p95 = {1000.0 / r.inter_completion.p95:.6f}"
            if r.inter_completion.p95 > 0 else " p95 = 0"
        )
        print(
            f" p99 = {1000.0 / r.inter_completion.p99:.6f}"
            if r.inter_completion.p99 > 0 else " p99 = 0"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_torch(args)

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    dataset_df = pd.read_csv(
        args.csv
    )

    if dataset_df.empty:
        raise RuntimeError("Dataset CSV is empty")

    model = create_model(
        args.weights
    )

    save_system_info(
        args,
        model,
    )

    print()
    print("=" * 72)
    print("HYPERSTARCOP CPU FP32 OPTIMIZED BENCHMARK")
    print("=" * 72)
    print("Batch: 1 ALWAYS")
    print("Profile:", args.profile)
    print("Dataset images:", len(dataset_df))
    print("Inference workers:", args.workers)
    print("Pre workers:", args.pre_workers)
    print("Post workers:", args.post_workers)
    print("Slots:", args.slots)
    print("PyTorch intra-op:", torch.get_num_threads())
    print("PyTorch inter-op:", torch.get_num_interop_threads())
    print("Pin:", args.pin)

    if (
        args.workers > 1
        and args.intra_threads > 1
    ):
        print()
        print(
            "WARNING: workers > 1 and intra-threads > 1 can oversubscribe CPU cores."
        )
        print(
            "For max throughput, start with --intra-threads 1."
        )

    if args.validate:
        print()
        print(">>> STAGE 1: VALIDATION")
        run_validation(
            args,
            dataset_df,
            model,
        )
        print(">>> VALIDATION FINISHED")

    results = []

    if args.profile in ("all", "baseline"):
        print()
        print(">>> STAGE 2: BASELINE BATCH-1")
        baseline_results = run_baseline(
            args,
            dataset_df,
            model,
        )
        results.extend(baseline_results)
        for _r in baseline_results:
            print_result(_r)
        print(">>> BASELINE FINISHED")

    if args.profile in ("all", "max-model-only"):
        print()
        print(">>> STAGE 3: MAX MODEL-ONLY THROUGHPUT")
        r = run_max_model_only(
            args,
            dataset_df,
            model,
        )
        results.append(r)
        print_result(r)
        print(">>> MAX MODEL-ONLY FINISHED")

    if args.profile in ("all", "max-e2e"):
        print()
        print(">>> STAGE 4: MAX END-TO-END THROUGHPUT")
        r = run_max_e2e(
            args,
            dataset_df,
            model,
        )
        results.append(r)
        print_result(r)
        print(">>> MAX END-TO-END FINISHED")

    save_results(
        args,
        results,
    )

    print()
    print("Results saved to:")
    print(args.out)


if __name__ == "__main__":
    main()
