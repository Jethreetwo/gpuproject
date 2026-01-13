from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

try:
    from task import input_t, output_t
except Exception:  # pragma: no cover
    input_t = Any
    output_t = Any

try:
    from torch._higher_order_ops.torchbind import call_torchbind_fake  # noqa: F401
except Exception:  # pragma: no cover
    call_torchbind_fake = None

try:
    import cuda.bindings.driver as cuda  # noqa: F401
except Exception:  # pragma: no cover
    cuda = None

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

# Kernel configuration parameters
mma_tiler_mnk = (128, 128, 256)
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
threads_per_cta = 128
num_acc_stage = 1
# Mainloop staging (TMA -> SMEM). Higher improves overlap but uses more SMEM.
num_ab_stage = 4
num_tmem_alloc_cols = 512


@dataclass(frozen=True)
class KernelConfig:
    mma_tiler_mnk: tuple[int, int, int] = mma_tiler_mnk
    mma_inst_shape_k: int = mma_inst_shape_k
    sf_vec_size: int = sf_vec_size
    threads_per_cta: int = threads_per_cta
    num_acc_stage: int = num_acc_stage
    num_ab_stage: int = num_ab_stage
    num_tmem_alloc_cols: int = num_tmem_alloc_cols


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b1: cute.CopyAtom,
    mB_nkl1: cute.Tensor,
    tma_atom_b2: cute.CopyAtom,
    mB_nkl2: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb1: cute.CopyAtom,
    mSFB_nkl1: cute.Tensor,
    tma_atom_sfb2: cute.CopyAtom,
    mSFB_nkl2: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    num_tma_load_bytes: cutlass.Constexpr[int],
    mma_tiler_mnk: cutlass.Constexpr = mma_tiler_mnk,
    mma_inst_shape_k: cutlass.Constexpr[int] = mma_inst_shape_k,
    sf_vec_size: cutlass.Constexpr[int] = sf_vec_size,
    threads_per_cta: cutlass.Constexpr[int] = threads_per_cta,
    num_acc_stage: cutlass.Constexpr[int] = num_acc_stage,
    num_ab_stage: cutlass.Constexpr[int] = num_ab_stage,
    num_tmem_alloc_cols: cutlass.Constexpr[int] = num_tmem_alloc_cols,
    epilogue_op: cutlass.Constexpr = lambda x: x
    * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx = cute.arch.thread_idx()

    bidx, bidy, bidz = cute.arch.block_idx()
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)

    cta_coord = (bidx, bidy, bidz)
    mma_tile_coord_mnl = (
        cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
        cta_coord[1],
        cta_coord[2],
    )
    @cute.struct
    class SharedStorage:
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_holding_buf: cutlass.Int32

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )
    sB1 = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    sB2 = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )
    sSFB1 = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )
    sSFB2 = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            threads_per_cta,
        ),
    ).make_participants()

    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gB_nkl1 = cute.local_tile(
        mB_nkl1, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gB_nkl2 = cute.local_tile(
        mB_nkl2, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl1 = cute.local_tile(
        mSFB_nkl1, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFB_nkl2 = cute.local_tile(
        mSFB_nkl2, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None, None)
    )
    k_tile_cnt = cute.size(gA_mkl, mode=[3])

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA_mkl)
    tCgB1 = thr_mma.partition_B(gB_nkl1)
    tCgB2 = thr_mma.partition_B(gB_nkl2)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    tCgSFB1 = thr_mma.partition_B(gSFB_nkl1)
    tCgSFB2 = thr_mma.partition_B(gSFB_nkl2)
    tCgC = thr_mma.partition_C(gC_mnl)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB1, tBgB1 = cpasync.tma_partition(
        tma_atom_b1,
        0,
        cute.make_layout(1),
        cute.group_modes(sB1, 0, 3),
        cute.group_modes(tCgB1, 0, 3),
    )
    tBsB2, tBgB2 = cpasync.tma_partition(
        tma_atom_b2,
        0,
        cute.make_layout(1),
        cute.group_modes(sB2, 0, 3),
        cute.group_modes(tCgB2, 0, 3),
    )
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3),
        cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    tBsSFB1, tBgSFB1 = cpasync.tma_partition(
        tma_atom_sfb1,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB1, 0, 3),
        cute.group_modes(tCgSFB1, 0, 3),
    )
    tBsSFB1 = cute.filter_zeros(tBsSFB1)
    tBgSFB1 = cute.filter_zeros(tBgSFB1)
    tBsSFB2, tBgSFB2 = cpasync.tma_partition(
        tma_atom_sfb2,
        0,
        cute.make_layout(1),
        cute.group_modes(sSFB2, 0, 3),
        cute.group_modes(tCgSFB2, 0, 3),
    )
    tBsSFB2 = cute.filter_zeros(tBsSFB2)
    tBgSFB2 = cute.filter_zeros(tBgSFB2)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB1 = tiled_mma.make_fragment_B(sB1)
    tCrB2 = tiled_mma.make_fragment_B(sB2)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    tmem.allocate(num_tmem_alloc_cols)
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc1 = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
    acc1_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc1)
    acc_tmem_ptr1 = cute.recast_ptr(
        acc_tmem_ptr + acc1_offset,
        dtype=cutlass.Float32,
    )
    tCtAcc2 = cute.make_tensor(acc_tmem_ptr1, tCtAcc_fake.layout)
    acc2_offset = tcgen05.find_tmem_tensor_col_offset(tCtAcc2)

    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    acc_tmem_base = acc_tmem_ptr + acc1_offset + acc2_offset
    sfa_tmem_ptr = cute.recast_ptr(acc_tmem_base, dtype=sf_dtype)
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    sfa_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
    sfb_tmem_ptr1 = cute.recast_ptr(
        acc_tmem_base + sfa_offset,
        dtype=sf_dtype,
    )
    tCtSFB1 = cute.make_tensor(sfb_tmem_ptr1, tCtSFB_layout)
    sfb_offset = tcgen05.find_tmem_tensor_col_offset(tCtSFB1)
    sfb_tmem_ptr2 = cute.recast_ptr(
        acc_tmem_base + sfa_offset + sfb_offset,
        dtype=sf_dtype,
    )
    tCtSFB2 = cute.make_tensor(sfb_tmem_ptr2, tCtSFB_layout)

    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
        sf_dtype,
    )

    tCsSFA_compact = cute.filter_zeros(sSFA)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
    )
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    tCsSFB1_compact = cute.filter_zeros(sSFB1)
    tCtSFB1_compact = cute.filter_zeros(tCtSFB1)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB1_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    tCsSFB1_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB1_compact)
    tCsSFB1_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB1_compact_s2t_
    )
    tCtSFB1_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB1_compact)

    tCsSFB2_compact = cute.filter_zeros(sSFB2)
    tCtSFB2_compact = cute.filter_zeros(tCtSFB2)
    tCsSFB2_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB2_compact)
    tCsSFB2_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy_s2t_sfb, tCsSFB2_compact_s2t_
    )
    tCtSFB2_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB2_compact)

    tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    tBgB1 = tBgB1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    tBgB2 = tBgB2[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    tBgSFB1 = tBgSFB1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    tBgSFB2 = tBgSFB2[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        num_kblocks = cute.size(tCrA, mode=[2])

        # Prologue: prime the pipeline leaving one free stage so we can prefetch
        # the next tile before compute without deadlocking the ring buffer.
        prefetch_tiles = num_ab_stage - 1
        if prefetch_tiles < 1:
            prefetch_tiles = 1
        if prefetch_tiles > k_tile_cnt:
            prefetch_tiles = k_tile_cnt
        for _ in range(prefetch_tiles):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b1,
                tBgB1[(None, ab_empty.count)],
                tBsB1[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b2,
                tBgB2[(None, ab_empty.count)],
                tBsB2[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, ab_empty.count)],
                tAsSFA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfb1,
                tBgSFB1[(None, ab_empty.count)],
                tBsSFB1[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_sfb2,
                tBgSFB2[(None, ab_empty.count)],
                tBsSFB2[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )

        for k_tile in range(k_tile_cnt):
            ab_full = ab_consumer.wait_and_advance()

            # Prefetch the next tile `prefetch_tiles` ahead to maximize overlap.
            next_tile = k_tile + prefetch_tiles
            if next_tile < k_tile_cnt:
                ab_empty = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b1,
                    tBgB1[(None, ab_empty.count)],
                    tBsB1[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_b2,
                    tBgB2[(None, ab_empty.count)],
                    tBsB2[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfb1,
                    tBgSFB1[(None, ab_empty.count)],
                    tBsSFB1[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )
                cute.copy(
                    tma_atom_sfb2,
                    tBgSFB2[(None, ab_empty.count)],
                    tBsSFB2[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                )

            s2t_stage_coord = (None, None, None, None, ab_full.index)
            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
            tCsSFB1_compact_s2t_staged = tCsSFB1_compact_s2t[s2t_stage_coord]
            tCsSFB2_compact_s2t_staged = tCsSFB2_compact_s2t[s2t_stage_coord]
            cute.copy(
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t_staged,
                tCtSFA_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB1_compact_s2t_staged,
                tCtSFB1_compact_s2t,
            )
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB2_compact_s2t_staged,
                tCtSFB2_compact_s2t,
            )

            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (
                    None,
                    None,
                    kblock_idx,
                    ab_full.index,
                )

                sf_kblock_coord = (None, None, kblock_idx)
                tiled_mma.set(
                    tcgen05.Field.SFA,
                    tCtSFA[sf_kblock_coord].iterator,
                )
                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB1[sf_kblock_coord].iterator,
                )
                cute.gemm(
                    tiled_mma,
                    tCtAcc1,
                    tCrA[kblock_coord],
                    tCrB1[kblock_coord],
                    tCtAcc1,
                )

                tiled_mma.set(
                    tcgen05.Field.SFB,
                    tCtSFB2[sf_kblock_coord].iterator,
                )
                cute.gemm(
                    tiled_mma,
                    tCtAcc2,
                    tCrA[kblock_coord],
                    tCrB2[kblock_coord],
                    tCtAcc2,
                )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            ab_full.release()
        acc_empty.commit()

    op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
    copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc1)
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc1 = thr_copy_t2r.partition_S(tCtAcc1)
    tTR_tAcc2 = thr_copy_t2r.partition_S(tCtAcc2)
    tTR_gC = thr_copy_t2r.partition_D(tCgC)
    tTR_rAcc1 = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
    )
    tTR_rAcc2 = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
    )
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC[None, None, None, None, 0, 0, 0].shape, c_dtype
    )
    simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype)
    tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

    acc_full = acc_consumer.wait_and_advance()

    cute.copy(tiled_copy_t2r, tTR_tAcc1, tTR_rAcc1)
    cute.copy(tiled_copy_t2r, tTR_tAcc2, tTR_rAcc2)

    acc_vec1 = epilogue_op(tTR_rAcc1.load())
    acc_vec2 = tTR_rAcc2.load()
    acc_vec = acc_vec1 * acc_vec2

    tTR_rC.store(acc_vec.to(c_dtype))
    cute.copy(simt_atom, tTR_rC, tTR_gC)

    acc_full.release()
    cute.arch.barrier()
    tmem.free(acc_tmem_ptr)
    return


def _make_my_kernel(config: KernelConfig):
    mma_tiler_mnk = config.mma_tiler_mnk
    mma_inst_shape_k = config.mma_inst_shape_k
    sf_vec_size = config.sf_vec_size
    threads_per_cta = config.threads_per_cta
    num_acc_stage = config.num_acc_stage
    num_ab_stage = config.num_ab_stage
    num_tmem_alloc_cols = config.num_tmem_alloc_cols

    @cute.jit
    def my_kernel(
        a_ptr: cute.Pointer,
        b1_ptr: cute.Pointer,
        b2_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb1_ptr: cute.Pointer,
        sfb2_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: tuple,
        epilogue_op: cutlass.Constexpr = lambda x: x
        * (1.0 / (1.0 + cute.math.exp(-x, fastmath=True))),
    ):
        m, n, k, l = problem_size

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor1 = cute.make_tensor(
            b1_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        b_tensor2 = cute.make_tensor(
            b2_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n)),
        )

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor1.shape, sf_vec_size
        )
        sfb_tensor1 = cute.make_tensor(sfb1_ptr, sfb_layout)
        sfb_tensor2 = cute.make_tensor(sfb2_ptr, sfb_layout)

        mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype,
            (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (tiled_mma.thr_id.shape,),
        )

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            ab_dtype,
            num_ab_stage,
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            ab_dtype,
            num_ab_stage,
        )
        sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            num_ab_stage,
        )
        sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            num_ab_stage,
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            a_tensor,
            a_smem_layout,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b1, tma_tensor_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            b_tensor1,
            b_smem_layout,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b2, tma_tensor_b2 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            b_tensor2,
            b_smem_layout,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfa_tensor,
            sfa_smem_layout,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb1, tma_tensor_sfb1 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfb_tensor1,
            sfb_smem_layout,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfb_tensor2,
            sfb_smem_layout,
            mma_tiler_mnk,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
        num_tma_load_bytes = (
            a_copy_size + b_copy_size * 2 + sfa_copy_size + sfb_copy_size * 2
        ) * atom_thr_size

        grid = (
            cute.ceil_div(c_tensor.shape[0], mma_tiler_mnk[0]),
            cute.ceil_div(c_tensor.shape[1], mma_tiler_mnk[1]),
            c_tensor.shape[2],
        )

        kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b1,
            tma_tensor_b1,
            tma_atom_b2,
            tma_tensor_b2,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb1,
            tma_tensor_sfb1,
            tma_atom_sfb2,
            tma_tensor_sfb2,
            c_tensor,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            num_tma_load_bytes,
            mma_tiler_mnk,
            mma_inst_shape_k,
            sf_vec_size,
            threads_per_cta,
            num_acc_stage,
            num_ab_stage,
            num_tmem_alloc_cols,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
        )
        return

    return my_kernel


_compiled_kernel_cache: dict[KernelConfig, Callable] = {}
_disabled_configs: set[KernelConfig] = set()
_selected_config: KernelConfig | None = None


def compile_kernel(config: KernelConfig) -> Callable:
    compiled = _compiled_kernel_cache.get(config)
    if compiled is not None:
        return compiled

    my_kernel = _make_my_kernel(config)

    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b1_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b2_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb1_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb2_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    compiled = cute.compile(
        my_kernel,
        a_ptr,
        b1_ptr,
        b2_ptr,
        sfa_ptr,
        sfb1_ptr,
        sfb2_ptr,
        c_ptr,
        (0, 0, 0, 0),
    )
    _compiled_kernel_cache[config] = compiled
    return compiled


def launch(
    compiled_kernel: Callable,
    a,
    b1,
    b2,
    sfa_permuted,
    sfb1_permuted,
    sfb2_permuted,
    c,
    problem_size: tuple[int, int, int, int],
):
    m, n, k, l = problem_size

    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b1_ptr = make_ptr(ab_dtype, b1.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b2_ptr = make_ptr(ab_dtype, b2.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(
        sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb1_ptr = make_ptr(
        sf_dtype, sfb1_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb2_ptr = make_ptr(
        sf_dtype, sfb2_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    compiled_kernel(
        a_ptr, b1_ptr, b2_ptr, sfa_ptr, sfb1_ptr, sfb2_ptr, c_ptr, (m, n, k, l)
    )
    return c


_CONFIG_STAGE4 = KernelConfig(mma_tiler_mnk=(128, 128, 256), num_ab_stage=4)
_CONFIG_STAGE3 = KernelConfig(mma_tiler_mnk=(128, 128, 256), num_ab_stage=3)


def custom_kernel(data: input_t) -> output_t:
    global _selected_config
    a, b1, b2, _, _, _, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data

    _, k_packed, _ = a.shape
    m, n, l = c.shape
    k = k_packed * 2

    if _selected_config is not None and _selected_config not in _disabled_configs:
        compiled = compile_kernel(_selected_config)
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

    for config in (_CONFIG_STAGE4, _CONFIG_STAGE3):
        if config in _disabled_configs:
            continue
        try:
            compiled = compile_kernel(config)
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
            _selected_config = config
            return c
        except Exception:
            _disabled_configs.add(config)
            continue

    raise RuntimeError("Failed to compile/launch any kernel configuration.")
