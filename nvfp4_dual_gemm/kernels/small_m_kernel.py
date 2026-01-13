from __future__ import annotations

from .base_kernel import KernelConfig, compile_kernel

_CONFIG = KernelConfig(mma_tiler_mnk=(64, 128, 256), num_ab_stage=3)


def compile(m: int, n: int, k: int):
    del m, n, k
    return compile_kernel(_CONFIG)

