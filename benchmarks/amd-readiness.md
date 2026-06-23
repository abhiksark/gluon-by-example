# AMD readiness audit

This repo's kernels and benchmark plumbing are written to run on AMD GPUs
(ROCm) with no infrastructure change, with one boundary: the Gluon twins stay
NVIDIA-only by construction. This document records what was checked on the
NVIDIA host so an AMD card "just works" when one is available. Performance and
ISA claims are deferred until they can be run on real AMD hardware.

## Device-neutral by construction

- PyTorch's ROCm build reuses the `torch.cuda` namespace. The input checks in
  `src/gluon_by_example/_validation.py` gate on `t.is_cuda`, which is True for
  AMD tensors on ROCm, so validation and the `@requires_cuda` tests run
  unchanged.
- `tools/bench_utils.py::gpu_slug()` derives the CSV and chart filename from
  `torch.cuda.get_device_name(0)`. On AMD this returns the AMD device name, so
  runs auto-produce `amd-...` CSVs and charts with no code change.
- `tools/bench_utils.py::device_arch()` reports the backend and version
  (`cuda <ver>` or `hip <ver>`), provenance the device slug cannot carry. Bench
  scripts can record it once AMD runs happen.

## What the device-string scan found

A scan for hardcoded device strings (`sm_NN`, compute-capability literals,
stray `cuda`/`nvidia` references) across `tools/` and `_validation.py` found no
string that breaks a ROCm run. Two spots are NVIDIA-flavored but safe:

- `triton_impl/matmul.py` autotune comment described only the CC 8.6 shared
  memory budget; it now points here and notes AMD wants its own config block.
- `tools/compare_gpus.py` hardcodes a color per known NVIDIA slug and a label
  helper that strips `nvidia-`/`geforce-`. An unknown AMD slug degrades
  gracefully: `GPU_COLORS.get(slug)` returns `None` (matplotlib auto-colors the
  line) and the label still renders (just unstripped, e.g. `AMD RADEON ...`).
  Add AMD entries to `GPU_COLORS` and the label stripper when AMD CSVs exist.

## Portability matrix (Triton kernels)

| Kernel | AMD status |
|---|---|
| `triton_impl/vector_add.py` | Ports clean. Bandwidth-bound, pure `tl.*`. |
| `triton_impl/softmax.py` | Ports clean. Bandwidth-bound, pure `tl.*`. |
| `triton_impl/normalization.py` | Ports clean. Bandwidth-bound, pure `tl.*`. |
| `triton_impl/scan.py` | Ports clean. Pure `tl.*`. |
| `triton_impl/matmul.py` | Ports with re-tuning. `tl.dot` lowers to AMD matrix cores; the autotune configs are NVIDIA-tuned and need an AMD config block (64-wide wavefront, LDS budget). |
| `triton_impl/attention.py` | Ports with re-tuning. Same as matmul: functional via `tl.dot`, autotune space needs an AMD block. |

The Gluon twins in `src/gluon_by_example/gluon_impl/` do not port: they use
NVIDIA-only constructs (`mma_v2`, `NVMMADistributedLayout`, `DotOperandLayout`,
`cp.async`). The AMD story is Triton-only by design.

## When an AMD card arrives

1. Pick the cheapest verifiable card: a consumer RDNA3 (7900 XTX, WMMA) locally,
   or a short MI300/MI210 cloud rental (MFMA) for the datacenter line.
2. Run `pytest tests/`. The four bandwidth-bound kernels should pass as-is;
   matmul and attention may need the AMD autotune block before they pass.
3. Run the per-chapter benches. `gpu_slug()` emits `amd-...` CSVs and charts
   automatically; have the bench record `device_arch()` for provenance.
4. Commit `benchmarks/results/*-amd-*.csv` and charts, marked AMD-verified,
   mirroring the existing A6000 commits.
5. No new kernel files are expected for the four simple kernels; matmul and
   attention get an AMD config block, not a rewrite.
