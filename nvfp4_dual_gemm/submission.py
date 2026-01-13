from task import input_t, output_t

from kernels import large_m_kernel, small_m_kernel
from kernels.base_kernel import launch

_cache = {}


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data

    _, k_packed, _ = a.shape
    m, n, l = c.shape
    k = k_packed * 2

    config_key = "small_m" if m <= 256 else "large_m"
    compiled = _cache.get(config_key)
    if compiled is None:
        if m <= 256:
            compiled = small_m_kernel.compile(m, n, k)
        else:
            compiled = large_m_kernel.compile(m, n, k)
        _cache[config_key] = compiled

    launch(
        compiled,
        a,
        b1,
        b2,
        sfa_permuted,
        sfb1_permuted,
        sfb2_permuted,
        c,
        (m, n, k, l),
    )
    return c

