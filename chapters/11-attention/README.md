# Chapter 11: FlashAttention in Triton

Self-attention is the operation that put GPU memory bandwidth on the critical path of large-model
inference. The naive implementation materializes an N-by-N score matrix that blows through HBM
bandwidth at quadratic cost. FlashAttention (Dao et al., 2022; FA2, 2023) avoids it entirely by
fusing the entire forward pass into a single kernel that never writes the score matrix to HBM.

Sources:
[Triton implementation](../../src/gluon_by_example/triton_impl/attention.py)

---

## What the kernel computes

Given Q, K, V of shape (Z, H, N, D), the forward pass computes:

```
S  = Q K^T / sqrt(D)        # (N, N) score matrix -- never materialized
P  = softmax(S, axis=-1)    # row-wise softmax
O  = P V                    # (N, D) output
```

In the causal (decoder) variant, positions j > i are masked to -inf before the softmax.

The wrapper flattens the batch and head dimensions so the inner kernels see (Z*H, N, D) tensors.
Inputs must be fp16 or bf16 on CUDA; D must be a power of two.

---

## No N-by-N matrix: online-softmax tiling

The central idea is that softmax does not need the full row to be computed at once. Given a
running maximum m and running sum l, a new block of scores (s_new) can update the result:

```
m_new  = max(m, max(s_new))
l_new  = l * exp(m - m_new) + sum(exp(s_new - m_new))
acc    = acc * exp(m - m_new) + exp(s_new - m_new) @ V_block
```

This is the online-softmax rescale from Milakov and Gimelshein (2018), applied block-by-block
across the KV dimension. At the end, `acc / l_new` is the correct output without ever having
stored S to global memory. The forward kernel maintains three running quantities (m_i, l_i, acc)
and loops over BLOCK_N-wide KV slices.

The two matrix multiplications inside the loop (QK^T and P @ V) are executed as tensor-core
`tl.dot` calls. Both contract over D=64 per tile. The BLOCK_M x BLOCK_N accumulator for QK^T
lives in registers; only the final output is written to HBM.

---

## Causal block skipping

For the causal case, each query tile at row index start_m only needs KV tiles with start_n <=
start_m * BLOCK_M. The kernel sets the KV loop upper bound to `(start_m + 1) * BLOCK_M` instead
of N. Blocks entirely above the diagonal are skipped, cutting compute to roughly half compared
with the full (non-causal) forward pass.

Within the last diagonal block (where both masked and unmasked positions appear) the kernel
applies the per-element mask `offs_m[:, None] >= offs_n[None, :]` before the softmax update,
zeroing out future positions.

---

## FA2 backward: the three-kernel split

The backward pass cannot reconstruct P cheaply from O alone; it would require recomputing the
softmax from scratch. FA2 (Dao, 2023) avoids storing P by saving the per-row log-sum-exp
L = m + log(l) from the forward and splitting the backward into three kernels.

**Step 1: preprocess.** One kernel computes `Delta[b, m] = rowsum(dO[b, m, :] * O[b, m, :])`.
Delta absorbs the softmax Jacobian's diagonal, enabling the subsequent kernels to compute dS
without holding P.

**Step 2: dk/dv.** One program per KV-block loops over all Q-blocks. For each Q-block it
reconstructs P^T from L, accumulates dV += P^T @ dO and dK += dS^T @ Q (where
dS = P * (dP - Delta) * sm_scale). No atomics are needed: each KV-block's dk and dv are fully
accumulated before being written once.

**Step 3: dq.** One program per Q-block loops over KV-blocks, accumulating
dQ += dS @ K for each block. Again atomic-free: each Q-block owns its dq accumulator.

The three-kernel split trades one extra kernel launch and the L buffer for freedom from atomic
collisions. All three backward kernels have the causal block-skip baked in symmetrically.

---

## Run it

```bash
pytest tests/test_attention.py -q
python chapters/11-attention/bench.py
```

---

## Benchmark

Benchmark results are pending a GPU run. The bench sweeps sequence length N in [512, 1024, 2048,
4096, 8192] at fixed Z=2, H=8, D=64, comparing `torch` (`F.scaled_dot_product_attention`) and
`triton` (the FA implementation in this chapter) for both causal and non-causal variants. Metric
is attention forward TFLOP/s charged at the full non-causal FLOPs (4ZHND^2 multiply-adds) so
causal and non-causal are comparable; causal will read lower because it skips half the KV blocks.

To generate results and chart on your GPU:

```bash
python chapters/11-attention/bench.py
```

*Written against Triton 3.7.0 (pip). API may move.*
