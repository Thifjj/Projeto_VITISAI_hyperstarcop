#!/bin/bash
set -e

'./hyperstarcop_zcu104_optimized' --profile max-e2e --model '/home/root/hyperstarcop.xmodel' --dataset '/home/root/STARCOP_mini' --csv '/home/root/STARCOP_mini/test_mini10.csv' --out '/home/root/hyperstarcop_sweep_results/best_reproduction' --runners 3 --pre-workers 4 --post-workers 16 --slots-per-runner 5 --iterations 500 --warmup 20 --pin --no-validate
