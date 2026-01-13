# NVFP4 Dual GEMM Competition Analysis & Strategy

## 1. WHAT THE KERNEL DOES (Exact Spec)

### Mathematical Operation
```
C = silu(A @ B1.T) * (A @ B2.T)

Where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

This is a **SwiGLU-style fused operation** - two parallel GEMMs with the same A matrix, followed by SiLU activation on the first result, then element-wise multiplication.

### Tensor Specifications

| Tensor | Shape | Dtype | Layout | Notes |
|--------|-------|-------|--------|-------|
| A | M × K × L | float4_e2m1fn (NVFP4) | K-major (stride: K, 1, M*K) | Shared by both GEMMs |
| B1 | N × K × L | float4_e2m1fn (NVFP4) | K-major (stride: K, 1, N*K) | First GEMM weight |
| B2 | N × K × L | float4_e2m1fn (NVFP4) | K-major (stride: K, 1, N*K) | Second GEMM weight |
| sfa | M × (K/16) × L | float8_e4m3fn | K-major | Block scales for A |
| sfb1 | N × (K/16) × L | float8_e4m3fn | K-major | Block scales for B1 |
| sfb2 | N × (K/16) × L | float8_e4m3fn | K-major | Block scales for B2 |
| C | M × N × L | float16 | N-major (stride: N, 1, M*N) | Output |

**Scale factor layout for kernel**: Permuted to `[32, 4, rest_M, 4, rest_K, L]` format per cuBLAS spec.

### Alignment/Divisibility Requirements
- M divisible by MMA tile M (128 in baseline)
- N divisible by MMA tile N (128 in baseline)
- K divisible by 256

### Benchmark Shapes (Ranking Determinants)

| M | N | K | L | SOL Target (μs) |
|---|---|---|---|-----------------|
| 256 | 4096 | 7168 | 1 | 4.708 |
| 512 | 4096 | 7168 | 1 | 8.714 |
| 256 | 3072 | 4096 | 1 | 2.125 |
| 512 | 3072 | 7168 | 1 | 6.535 |

**Scoring**: Geometric mean of benchmark times. Ranking by proximity to SOL.

### Validation
- `rtol=1e-03, atol=1e-03`
- 50 iterations per benchmark, CUDA events for timing
- L2 cache cleared between runs

---

## 2. SPEED-OF-LIGHT MODEL

### B200 Blackwell Specifications (at 1.5 GHz clock)

| Metric | Value | Notes |
|--------|-------|-------|
| FP4 TC Peak (dense) | ~9 PFLOPS | At boost; at 1.5 GHz ≈ 4.5 PFLOPS |
| HBM3e Bandwidth | 8 TB/s | 8 stacks × 1 TB/s each |
| L2 Cache | 64 MB | |
| SM Count | 168 | |
| Registers/SM | 256 KB | |
| Shared Mem/SM | 228 KB | |

**Clock adjustment**: Competition states 1.5 GHz clock. B200 boost is ~2.0-2.1 GHz.
- Adjusted FP4 peak: `9 PFLOPS × (1.5/2.0) ≈ 6.75 PFLOPS`

### Per-Testcase Analysis

For dual GEMM `C = silu(A @ B1) * (A @ B2)`:

**FLOPs Calculation**:
- Each GEMM: `2 × M × N × K` FLOPs (FP4→FP32 accumulation)
- Total for two GEMMs: `4 × M × N × K`
- SiLU + multiply epilogue: `~5 × M × N` (negligible vs GEMM)
- **Total**: `≈ 4 × M × N × K` FLOPs

**Bytes Moved** (unfused, cold cache):
- A: `M × K × 0.5` bytes (FP4 = 4 bits)
- B1: `N × K × 0.5` bytes
- B2: `N × K × 0.5` bytes
- sfa: `M × (K/16) × 1` bytes
- sfb1: `N × (K/16) × 1` bytes
- sfb2: `N × (K/16) × 1` bytes
- C: `M × N × 2` bytes (FP16 output)

**Total bytes**: `0.5×K×(M + 2N) + (K/16)×(M + 2N) + 2×M×N`
Simplified: `K×(M + 2N)×(0.5 + 1/16) + 2×M×N = K×(M + 2N)×0.5625 + 2×M×N`

### Speed-of-Light Table

| M | N | K | FLOPs (T) | Bytes (MB) | AI | Compute Bound (μs) | BW Bound (μs) | SOL (μs) | Given SOL |
|---|---|---|-----------|------------|----|--------------------|---------------|----------|-----------|
| 256 | 4096 | 7168 | 30.06 | 26.2 | 1147 | **4.45** | 3.28 | 4.45 | 4.708 |
| 512 | 4096 | 7168 | 60.13 | 30.5 | 1971 | **8.91** | 3.81 | 8.91 | 8.714 |
| 256 | 3072 | 4096 | 12.88 | 15.2 | 847 | **1.91** | 1.90 | 1.91 | 2.125 |
| 512 | 3072 | 7168 | 45.10 | 25.8 | 1748 | **6.68** | 3.22 | 6.68 | 6.535 |

**Calculations**:
```python
# Example for M=256, N=4096, K=7168:
flops = 4 * 256 * 4096 * 7168 = 30.06 TFLOPs
bytes = 7168*(256 + 2*4096)*0.5625 + 2*256*4096 = 26.2 MB
AI = flops / bytes = 1147 FLOPs/byte
compute_time = 30.06e12 / 6.75e15 = 4.45 μs
bw_time = 26.2e6 / 8e12 = 3.28 μs
```

### Key Insight: **ALL SHAPES ARE COMPUTE-BOUND**

Arithmetic intensities of 850-2000 FLOPs/byte far exceed the machine balance point (~843 FLOPs/byte for B200 FP4). The competition is purely about **maximizing Tensor Core utilization**.

The given SOL targets track very closely with the compute bound, suggesting:
1. The organizers' SOL is based on FP4 TC math throughput at 1.5 GHz
2. Achieving ~95%+ of TC peak is required to win

---

## 3. IMPLEMENTATION STRATEGY (Ranked Options)

### Path A: Top Performance (CUTLASS 3.x / CuTe DSL with Blackwell Optimizations)

**Target: 95%+ of TC peak**

#### Kernel Architecture

```
┌────────────────────────────────────────────────────────┐
│  Persistent CTA Kernel with Warpgroup Specialization   │
├────────────────────────────────────────────────────────┤
│  WarpGroup 0: Producer (TMA Loads)                     │
│  WarpGroup 1-3: Consumer (UMMA Compute)                │
├────────────────────────────────────────────────────────┤
│  Mainloop: Ping-pong pipeline (2-4 stages)             │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Stage 0: TMA load A, B1, B2, SFA, SFB1, SFB2     │  │
│  │ Stage 1: UMMA compute A@B1, A@B2 in parallel     │  │
│  │ Stage 2: (if needed) additional prefetch         │  │
│  └──────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────┤
│  Epilogue: Fused SiLU + Multiply (in registers)        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ acc1 = silu(acc1) // x * sigmoid(x)              │  │
│  │ result = acc1 * acc2                             │  │
│  │ Store result as FP16 via vectorized STG          │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

#### Critical Design Decisions

1. **Tile Sizes**:
   - For small M (256, 512): Use 64×256×256 or 128×128×256
   - Goal: Maximize K-dimension to amortize SMEM-to-TMEM copy latency
   - Trade-off: Larger K tile = more register pressure for scales

2. **Pipeline Stages**:
   - Current baseline: 1 stage (no overlap) - **SUBOPTIMAL**
   - Optimal: 2-3 stages for TMA/UMMA overlap
   - More stages = more SMEM usage, lower occupancy

3. **Dual GEMM Strategy**:
   - **Option A**: Interleaved K-blocks (current baseline)
     ```
     for k in K_tiles:
         acc1 += A[k] @ B1[k]  # shares A load
         acc2 += A[k] @ B2[k]
     ```
   - **Option B**: Split-accumulator with shared A in SMEM
     - Load A once per K-tile, use for both B1 and B2
     - Requires careful TMEM management (2× accumulator space)

4. **Scale Factor Handling**:
   - Baseline uses S2T (SMEM→TMEM) copy per K-block
   - Optimization: Prefetch scales into TMEM while previous MMA executes

5. **Epilogue Fusion**:
   - SiLU is `x * sigmoid(x)`
   - Compute in FP32 registers before FP16 conversion
   - Use fast math: `sigmoid(x) ≈ 1 / (1 + exp(-x))` with `__expf`

#### Specific Optimizations for Benchmark Shapes

| Shape | Strategy |
|-------|----------|
| M=256 | Use 64×N_tile or 128×N_tile; may benefit from split-K if N is large |
| M=512 | Standard 128×128×256 works well |
| Large K (7168) | 4 stages helps hide TMA latency; K_tile=256 means 28 iterations |
| Smaller K (4096) | 2 stages sufficient; fewer pipeline bubbles |

#### SMEM Layout & Swizzle

```
A tile: 128 × 256 × FP4 = 16 KB per stage (with swizzle)
B1 tile: 128 × 256 × FP4 = 16 KB per stage
B2 tile: 128 × 256 × FP4 = 16 KB per stage
SFA tile: 128 × 16 × FP8 = 2 KB per stage
SFB1 tile: 128 × 16 × FP8 = 2 KB per stage
SFB2 tile: 128 × 16 × FP8 = 2 KB per stage

Total per stage: ~54 KB
With 4 stages: ~216 KB (fits in 228 KB SMEM/SM)
```

**Swizzle pattern**: Use `Swizzle<3,4,3>` for 128-byte aligned FP4 tiles.

### Path B: Get-on-Board Fast (Tune Existing Baseline)

The provided `submission.py` is already a sophisticated CuTe kernel. Quick wins:

1. **Increase pipeline stages** (1 → 2 or 3):
   ```python
   num_ab_stage = 2  # Was 1
   num_acc_stage = 1  # Keep at 1
   ```

2. **Tune tile sizes for shapes**:
   ```python
   # For M=256 shapes:
   mma_tiler_mnk = (64, 128, 256)  # Smaller M tile
   # For M=512 shapes:
   mma_tiler_mnk = (128, 128, 256)  # Current default
   ```

3. **Add shape-specific dispatch**:
   ```python
   def custom_kernel(data):
       m, n, k, l = get_dims(data)
       if m <= 256:
           return kernel_small_m(data)
       else:
           return kernel_large_m(data)
   ```

4. **Enable persistent CTAs** (if not already):
   - Reuse CTA across multiple output tiles
   - Reduces kernel launch overhead and improves cache locality

---

## 4. BOTTLENECK ANALYSIS & NSIGHT COMPUTE METRICS

### Primary Metrics to Monitor

| Metric | Target | Indicates |
|--------|--------|-----------|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | >90% | TC utilization |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | <50% | Confirms compute-bound |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | >80% | Occupancy |
| `sm__sass_thread_inst_executed_op_dfma_pred_on.sum` | - | FP32 accumulate ops |
| `lts__t_sectors_srcunit_tex_op_read.sum` | Minimal | L2 cache misses |

### Nsight Compute Command
```bash
ncu --set full \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section SchedulerStats \
    --section WarpStateStats \
    --section InstructionStats \
    python eval.py benchmark benchmarks.txt
```

### Common Bottlenecks & Fixes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| TC util <80% | Pipeline bubbles | Increase stages, overlap TMA+MMA |
| TC util <80% | SMEM bank conflicts | Check swizzle pattern |
| TC util <80% | TMEM allocation stalls | Pre-allocate TMEM, reduce columns |
| TC util <80% | Scale factor loading | Prefetch SFA/SFB earlier |
| Memory BW >60% | Unfused ops | Ensure epilogue is fused |
| Low occupancy | Too much SMEM | Reduce stages or tile size |
| Warp stalls | Barrier waits | Check pipeline balance |

---

## 5. EXPERIMENT PLAN

### Step 1: Correctness Harness (Day 1)

```bash
# Set up environment
cd /path/to/nvfp4_dual_gemm
pip install nvidia-cutlass torch

# Run correctness tests
python -c "
from reference import generate_input, check_implementation, ref_kernel
from submission import custom_kernel
import torch

for m, n, k, l in [(256, 512, 256, 1), (1536, 512, 7168, 1)]:
    data = generate_input(m, n, k, l, seed=1111)
    out = custom_kernel(data)
    ref = ref_kernel(data)
    torch.cuda.synchronize()
    print(f'{m}x{n}x{k}: max_diff={torch.abs(out-ref).max().item():.6f}')
"
```

### Step 2: Baseline Profiling (Day 1-2)

```bash
# Benchmark current submission
python eval.py benchmark benchmarks.txt

# Profile with Nsight Compute (single shape)
echo "m: 256; n: 4096; k: 7168; l: 1; seed: 1111" > single.txt
ncu --target-processes all \
    --set full \
    -o baseline_profile \
    python eval.py profile single.txt

# Analyze
ncu-ui baseline_profile.ncu-rep
```

**Record baseline**:
- Time per shape
- TC utilization %
- Memory throughput %
- Warp stall reasons

### Step 3: Tuning Loop (Day 2-5)

```
┌─────────────────────────────────────────────────────┐
│                  TUNING DECISION TREE                │
├─────────────────────────────────────────────────────┤
│                                                      │
│  TC Util < 85%?                                      │
│  ├─ YES: Pipeline bubbles likely                     │
│  │   └─ Try: num_ab_stage = 2, 3, 4                 │
│  │   └─ Try: Prefetch distance tuning               │
│  │   └─ Check: S2T copy overlapping with MMA        │
│  │                                                   │
│  └─ NO (TC Util ≥ 85%):                             │
│      │                                               │
│      Occupancy < 2 waves?                            │
│      ├─ YES: SMEM/Register pressure                  │
│      │   └─ Try: Smaller tiles (64×128×256)         │
│      │   └─ Try: Fewer stages                        │
│      │                                               │
│      └─ NO: Diminishing returns territory            │
│          └─ Try: Persistent CTAs                     │
│          └─ Try: Split-K for small M                 │
│          └─ Try: Cluster launch (2×1×1)              │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Tuning parameters to sweep**:
```python
TILE_CONFIGS = [
    (64, 128, 256),
    (64, 256, 256),
    (128, 128, 256),
    (128, 256, 256),
]
STAGE_CONFIGS = [1, 2, 3, 4]
THREADS_CONFIGS = [128, 256]
```

### Step 4: Final-Mile Optimizations (Day 5-7)

1. **Persistent scheduling**:
   ```python
   # Instead of 1 tile per CTA, process multiple
   num_tiles_per_cta = min(4, total_tiles // num_sms)
   ```

2. **Alignment padding** (if shapes allow):
   ```python
   # Pad M to next multiple of 128 for better tile coverage
   m_padded = ((m + 127) // 128) * 128
   ```

3. **Prefetch tuning**:
   ```python
   # Experiment with cp.async commit/wait patterns
   prefetch_distance = 2  # Load 2 K-tiles ahead
   ```

4. **Register allocation**:
   - Reduce spills by limiting live values
   - Consider `--maxrregcount=128` for occupancy

### Plateau Decision Tree

```
IF results plateau at > 1.3× SOL:
  → Check for fundamental pipeline issue
  → Verify TMEM allocation isn't blocking
  → Consider rewriting from scratch with different approach

IF results plateau at 1.1-1.3× SOL:
  → Focus on epilogue fusion efficiency
  → Try cluster launch for better L2 locality
  → Experiment with different swizzle patterns

IF results plateau at < 1.1× SOL:
  → You're close! Micro-optimizations:
    - Instruction scheduling
    - Barrier placement
    - Memory access patterns
```

---

## 6. CODE SKELETON / FILE PLAN

### Directory Structure
```
nvfp4_dual_gemm/
├── submission.py          # Main entry point (required)
├── kernels/
│   ├── __init__.py
│   ├── base_kernel.py     # Shared kernel infrastructure
│   ├── small_m_kernel.py  # Optimized for M≤256
│   └── large_m_kernel.py  # Optimized for M>256
├── tuning/
│   ├── sweep.py           # Hyperparameter sweep
│   └── configs.py         # Tile/stage configurations
└── profiling/
    └── analyze.py         # Parse Nsight output
```

### Key Implementation Skeleton

```python
# submission.py - Shape-dispatching entry point
from task import input_t, output_t
from kernels import small_m_kernel, large_m_kernel

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa, sfb1, sfb2, c = data
    m, n, l = c.shape
    k = a.shape[1] * 2

    # Shape-specific dispatch
    config_key = (m <= 256, n >= 4096, k >= 4096)

    if config_key not in _cache:
        if m <= 256:
            _cache[config_key] = small_m_kernel.compile(m, n, k)
        else:
            _cache[config_key] = large_m_kernel.compile(m, n, k)

    kernel = _cache[config_key]
    kernel(a, b1, b2, sfa, sfb1, sfb2, c, (m, n, k, l))
    return c
```

```python
# kernels/base_kernel.py - Key optimizations over baseline

# 1. Increase pipeline depth
num_ab_stage = 3  # Was 1

# 2. Use optimal tile for shape
def get_tile_config(m, n, k):
    if m <= 256:
        return (64, 256, 256)  # Tall-thin: maximize N coverage
    elif n >= 4096:
        return (128, 256, 256)  # Wide: maximize N coverage
    else:
        return (128, 128, 256)  # Balanced

# 3. Prefetch pattern (pseudo-code)
@cute.kernel
def kernel(...):
    # Producer warp
    for k in range(k_tiles):
        # Issue TMA for k+2 while computing k
        if k + 2 < k_tiles:
            issue_tma_loads(k + 2)

        # Wait for k's data
        wait_for_tma(k)

        # Copy scales to TMEM (overlap with MMA if possible)
        copy_scales_s2t_async(k)

        # Compute both GEMMs for this K-tile
        for kblock in k_blocks:
            acc1 = mma(acc1, A[kblock], B1[kblock], sfa, sfb1)
            acc2 = mma(acc2, A[kblock], B2[kblock], sfa, sfb2)

    # Fused epilogue
    for elem in output_elements:
        val1 = silu(acc1[elem])
        result = val1 * acc2[elem]
        store_fp16(result)
```

### Build & Test Commands

```bash
# Install dependencies
pip install nvidia-cutlass>=3.5 torch>=2.4

# Run correctness tests
python -c "from submission import custom_kernel; print('Import OK')"

# Quick benchmark
python eval.py benchmark benchmarks.txt

# Full profiling
ncu -o profile python eval.py profile single.txt
```

---

## 7. SUMMARY: OPTIMAL PATH TO WIN

1. **Immediate wins** (hours):
   - Increase `num_ab_stage` from 1 to 2-3
   - Add shape-specific tile dispatch

2. **Medium effort** (days):
   - Implement proper pipeline with prefetching
   - Optimize scale factor loading (S2T overlap)
   - Tune per-shape configurations

3. **Final push** (days):
   - Persistent CTAs for reduced launch overhead
   - Cluster launch for L2 locality
   - Micro-tune instruction scheduling

**Expected outcome**: The given SOL targets are achievable. The baseline already uses the right primitives (CUTLASS 3.x, TMA, UMMA). The main gap is **pipeline depth and prefetching** - the current 1-stage design leaves significant performance on the table.

---

## 8. REFERENCE FILES

The reference implementation is located at:
- **Repository**: https://github.com/gpu-mode/reference-kernels
- **Path**: `problems/nvidia/nvfp4_dual_gemm/`

Key files:
- `task.yml` - Problem specification and benchmark shapes
- `reference.py` - PyTorch reference implementation and input generation
- `submission.py` - Current best-known CUTLASS/CuTe baseline
- `template.py` - Starter template for submissions
- `utils.py` - Validation utilities

---

## 9. SOURCES

- [NVIDIA B200 Datasheet](https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf)
- [Comparing Blackwell vs Hopper GPUs](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus)
- [NVIDIA DGX B200 Datasheet](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)
- [cuBLAS Block Scaling Factors Layout](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)
