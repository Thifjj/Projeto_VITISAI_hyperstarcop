#!/usr/bin/env python3
"""Search CPU FP32 throughput with the ZCU104 benchmark methodology.

The sweep keeps batch=1 and delegates every timed run to
``benchmark_hyperstarcop_cpu_optimized_v2.py``. It contains:

1. two-core comparisons (one model using two Torch threads versus two
   concurrent model copies using one thread each);
2. model-only scaling search;
3. end-to-end pipeline search;
4. long repeated validation of the best configuration in each mode.

Every run writes the same latency, throughput, stage-time and completion
cadence statistics used by the standalone CPU/ZCU104 comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    mode: str
    runners: int
    intra_threads: int
    pre_workers: int = 1
    post_workers: int = 1
    slots_per_runner: int = 1
    cpu_cores: int = 0
    pin: bool = False

    @property
    def total_slots(self) -> int:
        return self.runners * self.slots_per_runner

    def key(self) -> tuple:
        return (
            self.mode,
            self.runners,
            self.intra_threads,
            self.pre_workers,
            self.post_workers,
            self.slots_per_runner,
            self.cpu_cores,
            self.pin,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="HyperSTARCOP CPU FP32 throughput sweep"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--search-iterations", type=int, default=90)
    parser.add_argument("--final-iterations", type=int, default=500)
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--worker-warmup", type=int, default=3)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Repeat runs even when a valid summary already exists.",
    )
    args = parser.parse_args()

    for name in [
        "search_iterations",
        "final_iterations",
        "final_repeats",
    ]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")

    if args.warmup < 0 or args.worker_warmup < 0:
        raise ValueError("warmup values must be >= 0")

    args.root = args.root.resolve()
    args.out = (
        args.out.resolve()
        if args.out is not None
        else args.root / "resultados_cpu" / "hyperstarcop_cpu_sweep"
    )
    args.benchmark = (
        args.root / "scripts" / "benchmark_hyperstarcop_cpu_optimized_v2.py"
    )
    return args


def config_name(config: Config) -> str:
    pin = "pin" if config.pin else "nopin"
    return (
        f"{config.mode}__r{config.runners}_i{config.intra_threads}_"
        f"pre{config.pre_workers}_post{config.post_workers}_"
        f"spr{config.slots_per_runner}_c{config.cpu_cores}_{pin}"
    )


def read_summary(path: Path, mode: str):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    wanted = (
        "max_model_only_throughput"
        if mode == "model-only"
        else "max_end_to_end_throughput"
    )
    return next((row for row in rows if row["mode"] == wanted), None)


def run_config(
    args,
    config: Config,
    stage: str,
    iterations: int,
    repeat: int = 1,
):
    run_id = (
        f"{stage}__{config_name(config)}__rep{repeat}__n{iterations}"
    )
    run_dir = args.out / "runs" / run_id
    summary_path = run_dir / "benchmark_summary.csv"

    summary = None if args.rerun else read_summary(summary_path, config.mode)
    status = "reused" if summary is not None else "ok"

    if summary is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        profile = "max-model-only" if config.mode == "model-only" else "max-e2e"
        command = [
            sys.executable,
            str(args.benchmark),
            "--profile",
            profile,
            "--root",
            str(args.root),
            "--out",
            str(run_dir),
            "--workers",
            str(config.runners),
            "--pre-workers",
            str(config.pre_workers),
            "--post-workers",
            str(config.post_workers),
            "--slots",
            str(config.total_slots),
            "--intra-threads",
            str(config.intra_threads),
            "--interop-threads",
            "1",
            "--cpu-core-limit",
            str(config.cpu_cores),
            "--iterations",
            str(iterations),
            "--warmup",
            str(args.warmup),
            "--worker-warmup",
            str(args.worker_warmup),
            "--no-validate",
            "--pin" if config.pin else "--no-pin",
        ]

        (run_dir / "command.txt").write_text(
            " ".join(command) + "\n",
            encoding="utf-8",
        )
        (run_dir / "sweep_config.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"\n[{stage}] {run_id}", flush=True)
        with (run_dir / "execution.log").open(
            "w", encoding="utf-8"
        ) as log:
            process = subprocess.Popen(
                command,
                cwd=args.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                if "THROUGHPUT FPS" in line or "LATENCY ms" in line:
                    print(line.rstrip(), flush=True)
            return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"Benchmark failed with exit code {return_code}: {run_id}"
            )
        summary = read_summary(summary_path, config.mode)
        if summary is None:
            raise RuntimeError(f"Missing summary row: {summary_path}")
    else:
        print(f"\n[{stage}] reusing {run_id}", flush=True)

    record = {
        "stage": stage,
        "run_id": run_id,
        "repeat": repeat,
        "profile": config.mode,
        "runners": config.runners,
        "intra_threads": config.intra_threads,
        "interop_threads": 1,
        "pre_workers": config.pre_workers,
        "post_workers": config.post_workers,
        "slots_per_runner": config.slots_per_runner,
        "total_slots": config.total_slots,
        "cpu_cores": config.cpu_cores,
        "pin": config.pin,
        "iterations": iterations,
        "status": status,
        "result_dir": str(run_dir),
    }
    record.update(summary)
    return record


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_unique(target: list[Config], configs):
    keys = {config.key() for config in target}
    for config in configs:
        if config.key() not in keys:
            target.append(config)
            keys.add(config.key())


def throughput(row: dict) -> float:
    return float(row["throughput_fps"])


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    available = len(os.sched_getaffinity(0))
    max_cores = min(16, available)
    if max_cores < 2:
        raise RuntimeError("At least two logical CPUs are required")

    campaign = {
        "cpu": "AMD Ryzen 9 5980HX",
        "available_logical_cpus": available,
        "max_logical_cpus_used": max_cores,
        "batch": 1,
        "search_iterations": args.search_iterations,
        "final_iterations": args.final_iterations,
        "final_repeats": args.final_repeats,
        "measurement": (
            "throughput=completed/wall; latency includes queue and slot wait; "
            "same CPU benchmark implementation used for ZCU104 comparison"
        ),
    }
    (args.out / "campaign.json").write_text(
        json.dumps(campaign, indent=2) + "\n", encoding="utf-8"
    )

    all_rows = []

    # Two ways of using exactly two logical CPUs.
    two_core_configs = [
        Config("model-only", 1, 2, cpu_cores=2, pin=False),
        Config("model-only", 2, 1, cpu_cores=2, pin=True),
        Config(
            "end-to-end", 1, 2, pre_workers=1, post_workers=1,
            slots_per_runner=1, cpu_cores=2, pin=False,
        ),
        Config(
            "end-to-end", 2, 1, pre_workers=1, post_workers=1,
            slots_per_runner=1, cpu_cores=2, pin=True,
        ),
    ]
    for config in two_core_configs:
        all_rows.append(
            run_config(args, config, "10_two_core", args.search_iterations)
        )
        write_csv(args.out / "all_runs.csv", all_rows)

    # Model-only: scale either Torch threads in one model or independent models.
    model_configs = []
    for count in [1, 2, 4, 8, max_cores]:
        add_unique(
            model_configs,
            [Config("model-only", 1, count, cpu_cores=count, pin=False)],
        )
        add_unique(
            model_configs,
            [Config("model-only", count, 1, cpu_cores=count, pin=True)],
        )
    for workers, intra in [(2, 2), (4, 2), (8, 2)]:
        cores = min(max_cores, workers * intra)
        add_unique(
            model_configs,
            [Config("model-only", workers, intra, cpu_cores=cores, pin=False)],
        )

    model_rows = []
    for config in model_configs:
        row = run_config(
            args, config, "20_model_search", args.search_iterations
        )
        model_rows.append(row)
        all_rows.append(row)
        write_csv(args.out / "all_runs.csv", all_rows)

    best_model_row = max(model_rows, key=throughput)
    best_model_config = next(
        config for config in model_configs
        if config_name(config) in best_model_row["run_id"]
    )

    # End-to-end staged search, always allowed to use the whole notebook.
    e2e_configs = []
    for runners in [1, 2, 4, 8, max_cores]:
        add_unique(
            e2e_configs,
            [
                Config(
                    "end-to-end",
                    runners,
                    1,
                    pre_workers=2,
                    post_workers=1,
                    slots_per_runner=2,
                    cpu_cores=max_cores,
                    pin=True,
                )
            ],
        )

    initial_rows = []
    for config in e2e_configs:
        row = run_config(
            args, config, "30_e2e_runners", args.search_iterations
        )
        initial_rows.append(row)
        all_rows.append(row)
        write_csv(args.out / "all_runs.csv", all_rows)

    best_initial = max(initial_rows, key=throughput)
    best_runners = int(best_initial["runners"])

    refine_configs = []
    for pre_workers in [1, 2, 4, 8]:
        add_unique(
            refine_configs,
            [
                Config(
                    "end-to-end", best_runners, 1,
                    pre_workers=pre_workers, post_workers=1,
                    slots_per_runner=2, cpu_cores=max_cores, pin=True,
                )
            ],
        )
    for post_workers in [1, 2, 4]:
        add_unique(
            refine_configs,
            [
                Config(
                    "end-to-end", best_runners, 1,
                    pre_workers=2, post_workers=post_workers,
                    slots_per_runner=2, cpu_cores=max_cores, pin=True,
                )
            ],
        )
    for slots_per_runner in [1, 2, 4]:
        add_unique(
            refine_configs,
            [
                Config(
                    "end-to-end", best_runners, 1,
                    pre_workers=2, post_workers=1,
                    slots_per_runner=slots_per_runner,
                    cpu_cores=max_cores, pin=True,
                )
            ],
        )
    add_unique(
        refine_configs,
        [
            Config(
                "end-to-end", best_runners, 1,
                pre_workers=2, post_workers=1,
                slots_per_runner=2, cpu_cores=max_cores, pin=False,
            )
        ],
    )

    refine_rows = []
    for config in refine_configs:
        row = run_config(
            args, config, "40_e2e_refine", args.search_iterations
        )
        refine_rows.append(row)
        all_rows.append(row)
        write_csv(args.out / "all_runs.csv", all_rows)

    combined_e2e = initial_rows + refine_rows
    best_e2e_row = max(combined_e2e, key=throughput)
    all_e2e_configs = e2e_configs + refine_configs
    best_e2e_config = next(
        config for config in all_e2e_configs
        if config_name(config) in best_e2e_row["run_id"]
    )

    final_rows = []
    for config in [best_model_config, best_e2e_config]:
        for repeat in range(1, args.final_repeats + 1):
            row = run_config(
                args,
                config,
                "50_final",
                args.final_iterations,
                repeat,
            )
            final_rows.append(row)
            all_rows.append(row)
            write_csv(args.out / "all_runs.csv", all_rows)

    write_csv(
        args.out / "ranking_model_search.csv",
        sorted(model_rows, key=throughput, reverse=True),
    )
    write_csv(
        args.out / "ranking_e2e_search.csv",
        sorted(combined_e2e, key=throughput, reverse=True),
    )
    write_csv(
        args.out / "final_runs.csv",
        sorted(final_rows, key=lambda row: (row["profile"], row["repeat"])),
    )

    selected = {}
    for mode, config in [
        ("model-only", best_model_config),
        ("end-to-end", best_e2e_config),
    ]:
        rows = [
            row for row in final_rows
            if row["profile"] == mode
        ]
        fps = [throughput(row) for row in rows]
        p99 = [float(row["latency_p99_ms"]) for row in rows]
        selected[mode] = {
            "config": asdict(config),
            "total_slots": config.total_slots,
            "throughput_fps_mean": statistics.mean(fps),
            "throughput_fps_median": statistics.median(fps),
            "throughput_fps_min": min(fps),
            "throughput_fps_max": max(fps),
            "latency_p99_ms_mean": statistics.mean(p99),
            "repeats": len(rows),
            "iterations_per_repeat": args.final_iterations,
        }

    (args.out / "best_configs.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )

    print("\nCPU sweep completed", flush=True)
    print(json.dumps(selected, indent=2), flush=True)
    print(f"Results: {args.out}", flush=True)


if __name__ == "__main__":
    main()
