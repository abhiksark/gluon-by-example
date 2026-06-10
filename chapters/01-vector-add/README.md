# Chapter 1: Vector Add — What Even Is Gluon?

[Gluon](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon)
is a lower-level GPU language built on Triton's compiler stack
(`triton.experimental.gluon`). Same tile-based SPMD model — but where Triton
*infers* how tensor elements map onto threads, Gluon makes you *declare* it.

This chapter writes the simplest possible kernel both ways.

## The same kernel, twice

**Triton** ([source](../../src/gluon_by_example/triton_impl/vector_add.py)) —
the layout is invisible; the compiler picks it:

```python
offsets = pid * BLOCK + tl.arange(0, BLOCK)
```

**Gluon** ([source](../../src/gluon_by_example/gluon_impl/vector_add.py)) —
the layout is an explicit, first-class object:

```python
_LAYOUT = gl.BlockedLayout(
    size_per_thread=[8],   # each thread owns 8 contiguous elements
    threads_per_warp=[32],
    warps_per_cta=[4],     # 8 x 32 x 4 = 1024 = BLOCK
    order=[0],
)
offsets = pid * BLOCK + gl.arange(0, BLOCK, layout=_LAYOUT)
```

That `BlockedLayout` is the whole point. For vector add it buys nothing —
the kernel is bandwidth-bound and any sane layout saturates DRAM. But for
matmul and attention (chapters 5 and 7), controlling exactly which thread
holds which element is where the performance lives. Chapter 1 just makes the
mental model concrete while the kernel is trivial.

## Benchmark

![vector add bandwidth](../../benchmarks/charts/vector_add-nvidia-rtx-a6000.png)

All three implementations sit on top of each other at large sizes — as they
should. If your "faster" elementwise kernel beats `torch.add` by 2x, you are
probably measuring launch overhead, not bandwidth.

## Gotchas we hit

- The decorator is `from triton.experimental import gluon` → `@gluon.jit` —
  not under `gluon.language`.
- `@gluon.jit` functions must be defined in a real `.py` file. The JIT
  inspects source code, so kernels defined in a REPL or notebook cell fail
  with `OSError: could not get source code`.
- `gl.arange` requires the `layout=` argument. There is no default — that is
  the language working as designed.

## Run it

```bash
pytest tests/test_vector_add.py -v        # correctness, both backends
python chapters/01-vector-add/bench.py    # regenerate CSV + chart
```

*Written against Triton 3.7.0 (pip). Gluon is experimental; APIs move.*
