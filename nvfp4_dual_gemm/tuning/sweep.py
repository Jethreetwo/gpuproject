from __future__ import annotations

from kernels.base_kernel import KernelConfig, compile_kernel
from tuning.configs import STAGE_CONFIGS, THREADS_CONFIGS, TILE_CONFIGS


def main() -> None:
    configs: list[KernelConfig] = []
    for tile in TILE_CONFIGS:
        for stages in STAGE_CONFIGS:
            for threads in THREADS_CONFIGS:
                configs.append(
                    KernelConfig(
                        mma_tiler_mnk=tile,
                        num_ab_stage=stages,
                        threads_per_cta=threads,
                    )
                )

    print(f"{len(configs)} configs")
    for config in configs:
        print(config)

    # Example compile (requires CUDA + cutlass install):
    # compile_kernel(configs[0])


if __name__ == "__main__":
    main()

