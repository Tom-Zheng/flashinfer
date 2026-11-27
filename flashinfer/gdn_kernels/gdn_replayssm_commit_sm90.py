"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""SM90 ReplaySSM commit: persistent, TMA-pipelined, TF32 tensor cores.

Work item: one layer, one batch row and one value head.  Commit reads each
FP32 ``D x D`` checkpoint once and writes it back once in place, so it is a
streaming problem organized like the SM90 verify kernel: persistent CTAs (two
or three per SM, see ``_pipeline_config``) with three warp roles, and items
scheduled dynamically from a per-stream counter (see the producer) so the CTAs
finish together.  A producer warp streams items into an ``S``-stage
shared-memory ring with TMA (the raw K and V rows, and the checkpoint as one
box of 128 contiguous 128-byte rows, read front to back).  Two auxiliary warps take
alternate items and prepare everything that depends only on K and the gates.
One warpgroup (the base warps) does the checkpoint contractions.  Roles hand
off through mbarriers only.

The base warps copy ``h0`` into registers first and release the stage, so the
next loads never wait for the compute or the write-back.  The result is
staged in a separate tile with the same layout and leaves with one TMA store,
front to back.

With ``a`` accepted tokens, ``gamma_t = exp(L_t)`` the cumulative decay,
normalized keys ``k_t`` and ``r_t = h0 k_t``, the delta rule replayed from the
checkpoint ``h0`` gives per-token corrections

    c_t = beta_t (v_t - gamma_t r_t - sum_{j<t} gamma_t / gamma_j c_j k_j.k_t)

and ``h_a = gamma_{a-1} h0 + sum_{t<a} gamma_{a-1} / gamma_t c_t k_t^T``.  In
matrix form, with ``R = (V - (h0 K^T) diag(gamma)) diag(beta)`` (V x T),

    h_a = gamma_{a-1} h0 + (R P) K,
    P[j, t] = gamma_{a-1} / gamma_j N'[j, t]    (j <= t < a, else 0),
    N' = (I + L')^-1,   L'[j, i] = beta_i k_j.k_i   (j < i).

``N'`` depends only on the Gram matrix and beta, so no step divides by a
(possibly tiny) cumulative decay.  The auxiliary warps form the raw BF16 Gram
matrix ``G`` of the cached keys with ``mma`` (exact products), take the inverse
norms ``n`` from its diagonal and solve each column by back substitution on
``G`` directly: ``N'[j, t] = n_j Y[j, t]`` with

    Y[j, t] = delta_jt / n_j - sum_{j < i <= t} G[j, i] beta_i n_i^2 Y[i, t].

The base warps form ``h0 K^T`` with ``mma.sync`` straight from the ``h0``
registers: the wgmma accumulator layout they hold is, per warp, the
``mma.sync`` A fragment with the k index permuted, and ``K`` is read with the
same permutation.  ``R P`` follows with ``mma.sync`` on that accumulator
layout, and the rank-``a`` update is one TF32 ``wgmma`` chain whose
accumulator starts from ``gamma_{a-1} h0`` in FP32, with ``R P`` as the
register operand and ``K`` read from shared memory.

Precision: both products take the raw BF16 keys, which TF32 represents
exactly, and apply the inverse key norms in FP32 (to the columns of
``h0 K^T`` and of ``R P``).  ``R P`` and the update operand ``R P diag(n)``
are formed from two-term TF32 splits (``x = hi + lo``, dropping ``lo * lo``),
so apart from FP32 rounding the only TF32 error left is ``h0`` rounded to
nearest TF32 in ``h0 K^T`` (the SM100 tcgen05 commit truncates it); ``h0``
itself enters the result in FP32.

A tracked snapshot at step ``s < a`` is ``h_s`` from the same formula with
``P`` restricted to ``t <= s``.  The final state continues from it,
``h_a = gamma_{a-1} / gamma_s h_s + (R P_rest) K`` with ``P_rest`` restricted
to ``s < t < a``, so both results come from one accumulator.

Rejected tokens (``t >= a``) are selected out, never multiplied by zero: their
keys, gates, betas, Gram rows and columns of ``R`` are replaced by zeros, so
cached values past the accepted prefix cannot reach the checkpoint, even as
``0 * NaN``.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T as _T, dsl_user_op

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from . import _umma_helpers as U
from .cute_dsl_cache_naming import make_kernel_name
from .device_target import gdn_compile_options, gdn_device_target
from .gdn_decode_mtp_replayssm_sm90 import (
    _BOX,
    _D,
    _M_ROW,
    _SMEM_PER_SM,
    _TF32_HALF_ULP,
    _bits,
    _f32,
    _lds_u32,
    _mbar_wait,
    _ptx,
    _tf32,
    _tma_load,
    _tpad,
    _wgmma_wait,
)
import cuda.bindings.driver as cuda


_ACC = cutlass.Float32
_L2_EPS = 1.0e-6
_THREADS = 7 * 32  # one base warpgroup, two aux warps, one producer warp
HEAD_DIM = _D
MAX_WINDOW = 16


def _stage_bytes(T: int) -> int:
    # Checkpoint boxes, then K and V rows (Tp swizzled rows each).
    return _D * _D * 4 + 2 * _tpad(T) * _D * 2


def _work_bytes(T: int, nbuf: int, has_track: bool) -> int:
    tp = _tpad(T)
    keys = 2 * (2 * tp * 128 + _D * tp * 4)
    out = _D * _D * 4
    per_buf = (2 if has_track else 1) * 2 * tp * _M_ROW * 4 + 3 * 16 * 4
    per_aux = tp * (tp + 4) * 4 + 3 * 16 * 4
    return keys + out + nbuf * per_buf + 2 * per_aux + 2 * 16 * 4 + 1024


# Checkpoint ring depth.  Two stages saturate the stream (deeper rings are no
# faster), and the count must be even: the two aux warps take alternate items,
# so the previous use of a stage (and of an aux->base buffer) always belongs to
# the same aux warp and a parity wait never aliases a phase still in flight.
_STAGES = 2


def _pipeline_config(T: int, has_track: bool) -> tuple:
    """Pick (CTAs per SM, aux->base buffers).

    Independent CTAs overlap each other's per-item latency, so the most CTAs
    per SM that fit win: three for T <= 8, two (which fit every T <= 16)
    otherwise.
    """
    for ctas, nbuf in ((3, 4), (3, 2)):
        budget = _SMEM_PER_SM // ctas - 1024  # 1 KiB reserved per CTA
        need = _STAGES * (_stage_bytes(T) + 1024) + _work_bytes(T, nbuf, has_track)
        if need <= budget:
            return ctas, nbuf
    return 2, 4


@dsl_user_op
def _wgmma_tf32_rs(acc, a, b_desc, *, loc=None, ip=None):
    """``wgmma.mma_async`` m64nNk8 TF32, A from registers, B in shared memory.

    ``a`` holds this thread's four TF32 words of the 16 x 8 A slice of its
    warp (the ``mma.sync`` m16n8k8 A-fragment layout); the accumulators are
    tied in and out and always accumulate.
    """
    nreg = len(acc)
    regs = ",".join(f"${i}" for i in range(nreg))
    text = (
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, 1, 0;\n\t"
        f"wgmma.mma_async.sync.aligned.m64n{2 * nreg}k8.f32.tf32.tf32 "
        f"{{{regs}}}, {{${nreg},${nreg + 1},${nreg + 2},${nreg + 3}}}, "
        f"${nreg + 4}, p, 1, 1;\n\t}}"
    )
    constraints = (
        ",".join(["=f"] * nreg) + ",r,r,r,r,l," + ",".join(str(i) for i in range(nreg))
    )
    operands = (
        [Int32(w).ir_value(loc=loc, ip=ip) for w in a]
        + [Int64(b_desc).ir_value(loc=loc, ip=ip)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in acc]
    )
    st = llvm.StructType.get_literal([_T.f32()] * nreg)
    r = llvm.inline_asm(
        st,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(_T.f32(), r, [i], loc=loc, ip=ip))
        for i in range(nreg)
    ]


@dsl_user_op
def _lds_v2(addr: Int32, *, loc=None, ip=None):
    st = llvm.StructType.get_literal([_T.f32()] * 2)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0,$1}, [$2];",
        "=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(_T.f32(), r, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def _sts_v2(addr: Int32, x0, x1, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (x0, x1)],
        "st.shared.v2.f32 [$0], {$1,$2};",
        "r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _tf32_hi(x: Float32) -> Int32:
    """``x`` rounded to the nearest TF32, low mantissa bits cleared."""
    return (_bits(x) + _TF32_HALF_ULP) & Int32(-(1 << 13))


@cute.jit
def _swizzle(addr: Int32, mask: cutlass.Constexpr[int]) -> Int32:
    """Swizzle<log2(mask + 1), 4, 3> on a shared-memory byte address."""
    return addr ^ (((addr >> 7) & mask) << 4)


@cute.jit
def _kt_desc(addr: Int32, row_bytes: cutlass.Constexpr[int]) -> Int64:
    """wgmma descriptor of the K-major ``K^T`` operand: 8-row atoms of
    ``row_bytes``-byte rows (32- or 64-byte swizzle)."""
    mode = 2 if row_bytes == 64 else 3
    return (
        (Int64(addr) >> 4)
        | (Int64(1) << 16)
        | (Int64(8 * row_bytes >> 4) << 32)
        | (Int64(mode) << 62)
    )


@cute.jit
def _row_addr(base: Int32, v: Int32, k: Int32) -> Int32:
    """Byte address of checkpoint element (v, k) in a stage or staging tile:
    the checkpoint as 128 contiguous 128-byte rows (row 2 v + k // 32) in the
    128-byte swizzle, so one TMA box moves it front to back.  Accumulator
    fragment accesses (rows g, g + 8, column pairs 8 j + 2 c) are conflict
    free: two wavefronts per 256 bytes."""
    return _swizzle(base + (2 * v + k // _BOX) * 128 + (k % _BOX) * 4, 7)


@cute.jit
def _stage_and_store(
    res: cute.Tensor,
    out_base: Int32,
    row: Int32,
    c: Int32,
    is_issuer,
    tma_atom_st: cute.CopyAtom,
    s_out: cute.Tensor,
    tma_rows: cute.Tensor,
    dst: Int32,
):
    """Write the warpgroup's result (accumulator fragment: rows row, row + 8,
    column pairs 8 j + 2 c) into the staging tile and store it to checkpoint
    ``dst`` with one TMA box issued by warp 0."""
    # The previous store from the tile has read it.
    if is_issuer:
        cute.arch.cp_async_bulk_wait_group(0, read=True)
    cute.arch.barrier(barrier_id=1, number_of_threads=128)
    for j8 in cutlass.range_constexpr(_D // 8):
        for rh in cutlass.range_constexpr(2):
            _sts_v2(
                _row_addr(out_base, row + 8 * rh, 8 * j8 + 2 * c),
                res[4 * j8 + 2 * rh],
                res[4 * j8 + 2 * rh + 1],
            )
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier(barrier_id=1, number_of_threads=128)
    if is_issuer:
        U.simple_tma_copy(
            tma_atom_st, s_out, cute.local_tile(tma_rows, (2 * _D, _BOX), (dst, 0))
        )
        cute.arch.cp_async_bulk_commit_group()


@cute.kernel
def gdn_replayssm_commit_kernel_sm90(
    tma_atom: cute.CopyAtom,
    tma_atom_st: cute.CopyAtom,
    tma_rows: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_k: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    tma_v: cute.Tensor,
    log_g: cute.Tensor,
    beta: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    track_indices: cute.Tensor,
    track_steps: cute.Tensor,
    num_items: Int32,
    num_slots: Int32,
    batch: Int32,
    work: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int],
    NB: cutlass.Constexpr[int],
    NULL_BLOCK_ID: cutlass.Constexpr[int],
    HAS_TRACK: cutlass.Constexpr[bool],
):
    D = _D
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    base_warps = D // 16
    producer_warp = base_warps + 2
    Tp = _tpad(T)
    kt_tiles = Tp // 8
    halves = D // _BOX
    kt_row = Tp * 4  # bytes per K^T row
    kt_mask = 3 if kt_row == 64 else 1

    smem = utils.SmemAllocator()
    # Checkpoint stages and the result staging tile, both as 128 contiguous
    # 128-byte rows (see _row_addr).
    s_state = smem.allocate_tensor(
        _ACC,
        cute.make_layout((S, 2 * D, _BOX), stride=(D * D, _BOX, 1)),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    s_out = smem.allocate_tensor(
        _ACC,
        cute.make_layout((2 * D, _BOX), stride=(_BOX, 1)),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    # Raw K and V rows (128-byte swizzled); rows [T, Tp) are never loaded.
    s_k = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((S, Tp, D), stride=(Tp * D, D, 1)),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    s_v = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((S, Tp, D), stride=(Tp * D, D, 1)),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    # Keys as FP32 bits, exact in TF32 (double-buffered by item):
    #   s_kn  [buf, half, t, 32 words], K-major in the 128-byte swizzle (B of
    #         h0 K^T); raw (unnormalized) keys, rows t >= a are zero.
    #   s_knt [buf, k, Tp words] = K^T, K-major in the 32/64-byte swizzle (B
    #         of the rank update), columns permuted within each 8-token group
    #         so position p holds token 2p (p < 4) or 2 (p - 4) + 1, matching
    #         where the mma.sync C fragment of R P keeps them.
    s_kn = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout((2 * halves * Tp * _BOX,)), 1024
    )
    s_knt = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2 * D * Tp,)), 1024)
    s_full = smem.allocate_tensor(cutlass.Int64, cute.make_layout((S,)), 8)
    s_empty = smem.allocate_tensor(cutlass.Int64, cute.make_layout((S,)), 8)
    s_ready = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NB,)), 8)
    s_consumed = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NB,)), 8)
    # Per stage: flat checkpoint row (-1: nothing to commit), accepted length,
    # flat tracked row (-1: no snapshot) and tracked step.
    s_meta = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout((S, 5), stride=(5, 1)), 16
    )
    # P^T (and the remaining-step P^T of a snapshot) as TF32 hi/lo bits:
    # s_p[buf, part, t, j] = P[j, t].
    p_layout = cute.make_layout(
        (NB, 2, Tp, _M_ROW), stride=(2 * Tp * _M_ROW, Tp * _M_ROW, _M_ROW, 1)
    )
    s_p = smem.allocate_tensor(cutlass.Int32, p_layout, 16)
    s_pt = s_p
    if cutlass.const_expr(HAS_TRACK):
        s_pt = smem.allocate_tensor(cutlass.Int32, p_layout, 16)
    buf_layout = cute.make_layout((NB, 16), stride=(16, 1))
    s_gam = smem.allocate_tensor(_ACC, buf_layout, 16)
    s_beta = smem.allocate_tensor(_ACC, buf_layout, 16)
    # [buf, 0]: scale of h0 in the first result (gamma_{a-1}, or gamma_s with
    # a snapshot); [buf, 1]: gamma_{a-1} / gamma_s.
    s_scale = smem.allocate_tensor(_ACC, buf_layout, 16)
    # Aux-private scratch: cumulative log decay, beta n^2, inverse key norms
    # and the raw Gram matrix.
    aux_layout = cute.make_layout((2, 16), stride=(16, 1))
    s_lam = smem.allocate_tensor(_ACC, aux_layout, 16)
    s_betap = smem.allocate_tensor(_ACC, aux_layout, 16)
    s_inv = smem.allocate_tensor(_ACC, aux_layout, 16)
    # Base: inverse key norms of the item (double-buffered like s_kn).
    s_inv_b = smem.allocate_tensor(_ACC, aux_layout, 16)
    s_g = smem.allocate_tensor(
        _ACC,
        cute.make_layout((2, Tp, Tp + 4), stride=(Tp * (Tp + 4), Tp + 4, 1)),
        16,
    )

    if tidx == 0:
        for s in cutlass.range_constexpr(S):
            cute.arch.mbarrier_init(s_full.iterator + s, 1)
            cute.arch.mbarrier_init(s_empty.iterator + s, base_warps + 1)
        for s in cutlass.range_constexpr(NB):
            cute.arch.mbarrier_init(s_ready.iterator + s, 1)
            cute.arch.mbarrier_init(s_consumed.iterator + s, base_warps)
    cute.arch.mbarrier_init_fence()
    kn_words = 2 * halves * Tp * _BOX
    for i in cutlass.range_constexpr((kn_words + _THREADS - 1) // _THREADS):
        linear = tidx + i * _THREADS
        if linear < kn_words:
            s_kn[linear] = 0
    # Inverse norms of tokens [T, Tp) scale all-zero columns; keep them finite.
    if tidx < 2 * 16:
        s_inv_b[(tidx // 16, tidx % 16)] = cutlass.Float32(0.0)
    cute.arch.barrier()

    k_addr = s_k.iterator.toint()
    row_bytes = Tp * D * 2
    kn_addr = s_kn.iterator.toint()
    knt_addr = s_knt.iterator.toint()
    state_addr = s_state.iterator.toint()

    # ------------------------------------------------------------------ producer
    if warp == producer_warp:
        tx_bytes = D * D * 4 + 2 * T * D * 2
        # Checkpoint loads keep the default L2 policy: each line is written
        # back shortly after it is read, and an evict-first hint measured
        # slower.
        local = Int32(0)
        # Dynamic schedule: each CTA takes items bid and bid + grid, then the
        # next unclaimed items from a global counter, so CTAs the memory system
        # serves faster take more items and all finish together.  The static
        # start keeps small launches free of counter traffic.  The claim for
        # the next item and its request metadata are in flight while this
        # item is issued; the item reaches the other roles in s_meta.
        item = bid
        n_next = (bid // HV) % batch
        next_slot = state_indices[n_next]
        next_len = accept_lens[n_next]
        next_tslot = Int32(-1)
        next_tstep = Int32(-1)
        if cutlass.const_expr(HAS_TRACK):
            next_tslot = track_indices[n_next]
            next_tstep = track_steps[n_next]
        while item < num_items:
            stage = local % S
            claim = Int32(0)
            if lane == 0 and local > 0:
                claim = cute.arch.atomic_add(work.iterator.llvm_ptr, Int32(1))
            slot = next_slot
            n_commit = next_len
            tslot = next_tslot
            tstep = next_tstep
            if local >= S:
                _mbar_wait(s_empty.iterator + stage, ((local // S) - 1) % 2)
            layer = item // (HV * batch)
            hv = item % HV
            i_h = hv // (HV // H)
            if n_commit > T:
                n_commit = Int32(T)
            live = slot > NULL_BLOCK_ID and n_commit > 0
            flat = Int32(-1)
            if live:
                flat = layer * num_slots + slot
            tflat = Int32(-1)
            if cutlass.const_expr(HAS_TRACK):
                if live and tslot > NULL_BLOCK_ID and tstep >= 0 and tstep < n_commit:
                    tflat = layer * num_slots + tslot
            full = s_full.iterator + stage
            if lane == 0:
                s_meta[(stage, 0)] = flat
                s_meta[(stage, 1)] = n_commit
                s_meta[(stage, 2)] = tflat
                s_meta[(stage, 3)] = tstep
                s_meta[(stage, 4)] = item
                if live:
                    cute.arch.mbarrier_arrive_and_expect_tx(full, tx_bytes)
                else:
                    cute.arch.mbarrier_arrive(full)
            cute.arch.sync_warp()
            if live:
                _tma_load(
                    tma_atom,
                    cute.local_tile(tma_rows, (2 * D, _BOX), (flat * HV + hv, 0)),
                    s_state[(stage, None, None)],
                    full,
                    None,
                )
                _tma_load(
                    tma_atom_k,
                    cute.local_tile(tma_k, (1, T, D), (flat * H + i_h, 0, 0))[
                        (0, None, None)
                    ],
                    cute.local_tile(s_k[(stage, None, None)], (T, D), (0, 0)),
                    full,
                    None,
                )
                _tma_load(
                    tma_atom_v,
                    cute.local_tile(tma_v, (1, T, D), (flat * HV + hv, 0, 0))[
                        (0, None, None)
                    ],
                    cute.local_tile(s_v[(stage, None, None)], (T, D), (0, 0)),
                    full,
                    None,
                )
            local = local + 1
            item = bid + grid
            if local > 1:
                item = 2 * grid + cute.arch.shuffle_sync(claim, 0)
            if item < num_items:
                n_next = (item // HV) % batch
                next_slot = state_indices[n_next]
                next_len = accept_lens[n_next]
                if cutlass.const_expr(HAS_TRACK):
                    next_tslot = track_indices[n_next]
                    next_tstep = track_steps[n_next]
        # The last CTA to finish claiming resets the counters for the next
        # launch (graph replays included).
        if lane == 0:
            done = cute.arch.atomic_add(
                (work.iterator + 1).llvm_ptr, Int32(1), sem="acq_rel", scope="gpu"
            )
            if done == grid - 1:
                work[0] = Int32(0)
                work[1] = Int32(0)
        # Two end markers: one for each aux warp's parity (the base warps
        # stop at the first).
        for _ in cutlass.range_constexpr(2):
            stage = local % S
            if local >= S:
                _mbar_wait(s_empty.iterator + stage, ((local // S) - 1) % 2)
            if lane == 0:
                s_meta[(stage, 4)] = Int32(-1)
                cute.arch.mbarrier_arrive(s_full.iterator + stage)
            cute.arch.sync_warp()
            local = local + 1

    hmma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaTF32Op((16, 8, 8)),
        cute.make_layout((1, 1, 1)),
    )
    g = lane // 4
    c = lane % 4

    # ------------------------------------------------------------------ aux
    if warp >= base_warps and warp < producer_warp:
        aux = warp - base_warps
        local = aux
        go = Int32(1)
        while go > 0:
            stage = local % S
            buf = local % NB
            _mbar_wait(s_full.iterator + stage, (local // S) % 2)
            item = s_meta[(stage, 4)]
            if item < 0:
                go = Int32(0)
            else:
                hv = item % HV
                flat = s_meta[(stage, 0)]
                n_commit = s_meta[(stage, 1)]
                tflat = s_meta[(stage, 2)]
                tstep = s_meta[(stage, 3)]
                live = flat >= 0
                if local >= NB:
                    _mbar_wait(s_consumed.iterator + buf, ((local // NB) - 1) % 2)
                # Gates of the accepted tokens and their cumulative sum: lane t.
                t = lane
                log_decay = cutlass.Float32(0.0)
                beta_value = cutlass.Float32(0.0)
                if live and t < n_commit:
                    gate_idx = (Int64(flat) * HV + hv) * T + t
                    log_decay = log_g[gate_idx]
                    beta_value = beta[gate_idx]
                lam = log_decay
                for offset in [1, 2, 4, 8]:
                    other = cute.arch.shuffle_sync_up(
                        lam, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    if t >= offset:
                        lam = lam + other

                # Raw Gram K K^T with BF16 m16n8k16 (exact products).  Under the
                # 128-byte swizzle lane (g, c) reads word c of chunk (2 ks + m) ^ g
                # of its row, which is conflict free; rows >= a read as zero.
                k_stage = k_addr + stage * row_bytes
                for mt in cutlass.range_constexpr((Tp + 15) // 16):
                    a_rows = []
                    a_ok = []
                    for rh in cutlass.range_constexpr(2):
                        r = 16 * mt + g + 8 * rh
                        if cutlass.const_expr(16 * mt + 8 * rh >= Tp):
                            r = g
                        a_rows.append(k_stage + r * (D * 2))
                        a_ok.append(r < n_commit and 16 * mt + 8 * rh < Tp)
                    a_w = []
                    for ks in cutlass.range_constexpr(D // 16):
                        for m in cutlass.range_constexpr(2):
                            for rh in cutlass.range_constexpr(2):
                                w = _lds_u32(
                                    a_rows[rh] + (((2 * ks + m) ^ g) * 16) + c * 4
                                )
                                if not a_ok[rh]:
                                    w = Int32(0)
                                a_w.append(w)
                    for nt in cutlass.range_constexpr(kt_tiles):
                        b_r = 8 * nt + g
                        b_row = k_stage + b_r * (D * 2)
                        b_ok = b_r < n_commit
                        gacc = [cutlass.Float32(0.0)] * 4
                        for ks in cutlass.range_constexpr(D // 16):
                            b_w = []
                            for m in cutlass.range_constexpr(2):
                                w = _lds_u32(b_row + (((2 * ks + m) ^ g) * 16) + c * 4)
                                if not b_ok:
                                    w = Int32(0)
                                b_w.append(w)
                            base = 4 * ks
                            gacc = U.mma_bf16_16816(
                                a_w[base],
                                a_w[base + 1],
                                a_w[base + 2],
                                a_w[base + 3],
                                b_w[0],
                                b_w[1],
                                gacc,
                            )
                        for hs in cutlass.range_constexpr(2):
                            for rh in cutlass.range_constexpr(2):
                                if cutlass.const_expr(16 * mt + 8 * rh < Tp):
                                    s_g[
                                        (aux, 16 * mt + g + 8 * rh, 8 * nt + 2 * c + hs)
                                    ] = gacc[hs + 2 * rh]
                cute.arch.sync_warp()
                # The stage's K rows are no longer needed by this warp.
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_empty.iterator + stage)
                if t < Tp:
                    inv_k = cute.rsqrt(s_g[(aux, t, t)] + _L2_EPS, fastmath=True)
                    s_inv[(aux, t)] = inv_k
                    s_lam[(aux, t)] = lam
                    s_betap[(aux, t)] = beta_value * inv_k * inv_k
                    s_gam[(buf, t)] = cute.exp(lam, fastmath=True)
                    s_beta[(buf, t)] = beta_value
                cute.arch.sync_warp()

                # Column t of Y per lane: back substitution on the raw Gram matrix
                # (see the module docstring); Gram rows and per-token factors are
                # read with 16-byte loads.
                if t < Tp:
                    g_aux = s_g.iterator.toint() + aux * (Tp * (Tp + 4) * 4)
                    betap = []
                    lams = []
                    invs = []
                    for jq in cutlass.range_constexpr(Tp // 4):
                        betap += U.lds_v4(
                            s_betap.iterator.toint() + (aux * 16 + 4 * jq) * 4
                        )
                        lams += U.lds_v4(
                            s_lam.iterator.toint() + (aux * 16 + 4 * jq) * 4
                        )
                        invs += U.lds_v4(
                            s_inv.iterator.toint() + (aux * 16 + 4 * jq) * 4
                        )
                    diag = cutlass.Float32(1.0) / s_inv[(aux, t)]
                    ysol = []
                    for j in cutlass.range_constexpr(Tp):
                        init = diag
                        if j != t or t >= T:
                            init = cutlass.Float32(0.0)
                        ysol.append(init)
                    for ii in cutlass.range_constexpr(Tp):
                        i = Tp - 1 - ii
                        coef = _f32(Int32(betap[i])) * ysol[i]
                        for jq in cutlass.range_constexpr((i + 3) // 4):
                            w4 = U.lds_v4(g_aux + (i * (Tp + 4) + 4 * jq) * 4)
                            for e in cutlass.range_constexpr(4):
                                j = 4 * jq + e
                                if cutlass.const_expr(j < i):
                                    ysol[j] = ysol[j] - _f32(Int32(w4[e])) * coef
                    # With a snapshot at step s the first result covers t <= s and
                    # the second s < t < a; otherwise the first covers t < a.
                    last = n_commit - 1
                    if last < 0:
                        last = Int32(0)
                    first_end = last
                    if cutlass.const_expr(HAS_TRACK):
                        if tflat >= 0:
                            first_end = tstep
                    lam_end = s_lam[(aux, first_end)]
                    lam_last = s_lam[(aux, last)]
                    # [first or rest][hi or lo] TF32 words of row t of P^T.
                    words: list = [[[], []], [[], []]]
                    for j in cutlass.range_constexpr(Tp):
                        n_y = _f32(Int32(invs[j])) * ysol[j]
                        lam_j = _f32(Int32(lams[j]))
                        val = cute.exp(lam_end - lam_j, fastmath=True) * n_y
                        if j > t or t > first_end:
                            val = cutlass.Float32(0.0)
                        hi = _tf32_hi(val)
                        words[0][0].append(hi)
                        words[0][1].append(_tf32(val - _f32(hi)))
                        if cutlass.const_expr(HAS_TRACK):
                            val2 = cute.exp(lam_last - lam_j, fastmath=True) * n_y
                            if j > t or t <= first_end or t > last:
                                val2 = cutlass.Float32(0.0)
                            hi2 = _tf32_hi(val2)
                            words[1][0].append(hi2)
                            words[1][1].append(_tf32(val2 - _f32(hi2)))
                    for which in cutlass.range_constexpr(2 if HAS_TRACK else 1):
                        p_base = [s_p, s_pt][which].iterator.toint()
                        for part in cutlass.range_constexpr(2):
                            p_row = p_base + (((buf * 2 + part) * Tp + t) * _M_ROW) * 4
                            pw = words[which][part]
                            for jq in cutlass.range_constexpr(Tp // 4):
                                U.sts_v4(
                                    p_row + 16 * jq,
                                    pw[4 * jq],
                                    pw[4 * jq + 1],
                                    pw[4 * jq + 2],
                                    pw[4 * jq + 3],
                                )
                    if t == 0:
                        s_scale[(buf, 0)] = cute.exp(lam_end, fastmath=True)
                        s_scale[(buf, 1)] = cute.exp(lam_last - lam_end, fastmath=True)
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_ready.iterator + buf)
                local = local + 2

    # ------------------------------------------------------------------ base
    if warp < base_warps:
        row_base = warp * 16
        is_issuer = warp == 0
        out_base = s_out.iterator.toint()
        acc_r = cute.make_rmem_tensor((Tp // 2,), _ACC)
        v_r = cute.make_rmem_tensor((4 * kt_tiles,), _ACC)
        w_r = cute.make_rmem_tensor(
            (2 if HAS_TRACK else 1, kt_tiles, 2, 4), cutlass.Int32
        )
        res = cute.make_rmem_tensor((D // 2,), _ACC)
        local = Int32(0)
        go = Int32(1)
        while go > 0:
            stage = local % S
            buf = local % NB
            _mbar_wait(s_full.iterator + stage, (local // S) % 2)
            item = s_meta[(stage, 4)]
            if item < 0:
                go = Int32(0)
            else:
                hv = item % HV
                flat = s_meta[(stage, 0)]
                n_commit = s_meta[(stage, 1)]
                tflat = s_meta[(stage, 2)]
                live = flat >= 0
                kn_buf = kn_addr + (local % 2) * (halves * Tp * 128)
                knt_buf = knt_addr + (local % 2) * (D * Tp * 4)
                stage_state = state_addr + stage * (D * D * 4)

                if live:
                    # This warp's rows of h0, in the accumulator layout of the
                    # rank update, kept in FP32 registers.
                    for j8 in cutlass.range_constexpr(D // 8):
                        for rh in cutlass.range_constexpr(2):
                            pair = _lds_v2(
                                _row_addr(
                                    stage_state, row_base + g + 8 * rh, 8 * j8 + 2 * c
                                )
                            )
                            res[4 * j8 + 2 * rh] = pair[0]
                            res[4 * j8 + 2 * rh + 1] = pair[1]
                    # V values this lane needs for R.
                    for kk in cutlass.range_constexpr(kt_tiles):
                        for hs in cutlass.range_constexpr(2):
                            j_safe = 8 * kk + 2 * c + hs
                            if j_safe >= T:
                                j_safe = T - 1
                            for rh in cutlass.range_constexpr(2):
                                v_r[(kk * 2 + hs) * 2 + rh] = cutlass.Float32(
                                    s_v[(stage, j_safe, row_base + g + 8 * rh)]
                                )
                    # Keys into s_kn and their inverse norms into s_inv_b: 8 lanes
                    # per row, one 16-byte chunk (8 BF16) each; the raw rows are
                    # swizzled (chunk XOR row % 8) and so is s_kn (16-byte chunks
                    # 2 (sub % 4) + m of half sub // 4, XOR row % 8).  Rows t >= a
                    # are zeroed.
                    r = tidx // 8
                    sub = tidx % 8
                    row_ok = r < T
                    rr = r
                    if not row_ok:
                        rr = Int32(0)
                    w4 = U.lds_v4(
                        k_addr
                        + stage * row_bytes
                        + rr * (D * 2)
                        + ((sub ^ (rr % 8)) * 16)
                    )
                    xs = []
                    ss = cutlass.Float32(0.0)
                    for w in w4:
                        lo = U.bf16x2_lo_f32(w)
                        hi = U.bf16x2_hi_f32(w)
                        xs.append(lo)
                        xs.append(hi)
                        ss = ss + lo * lo + hi * hi
                    for offset in [4, 2, 1]:
                        ss = ss + cute.arch.shuffle_sync_bfly(
                            ss, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    inv = cute.rsqrt(ss + _L2_EPS, fastmath=True)
                    key_words = [_bits(x) for x in xs]
                    if r >= n_commit:
                        inv = cutlass.Float32(0.0)
                        key_words = [Int32(0)] * 8
                    if row_ok:
                        if sub == 0:
                            s_inv_b[(local % 2, rr)] = inv
                        row_addr = kn_buf + ((sub // 4) * Tp + rr) * 128
                        for m in cutlass.range_constexpr(2):
                            U.sts_v4(
                                row_addr + (((2 * (sub % 4) + m) ^ (rr % 8)) * 16),
                                key_words[4 * m],
                                key_words[4 * m + 1],
                                key_words[4 * m + 2],
                                key_words[4 * m + 3],
                            )

                # Everything this warp needs from the stage is in registers.
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_empty.iterator + stage)

                if live:
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    # h0 K^T with mma.sync from the h0 registers (rounded to
                    # nearest TF32) against the raw keys (exact in TF32).  The
                    # accumulator fragment of column pair 8 j + 2 c is the A
                    # fragment of k-step j with k permuted (slot c holds column
                    # 8 j + 2 c, slot c + 4 column 8 j + 2 c + 1), so the B rows
                    # are read with the same permutation.  Two accumulators per
                    # token tile (even / odd k-steps) halve the mma chain.
                    r_acc = [
                        hmma.make_fragment_C(hmma.partition_shape_C((16, 8)))
                        for _ in range(2 * kt_tiles)
                    ]
                    for i in cutlass.range_constexpr(2 * kt_tiles):
                        r_acc[i].fill(0.0)
                    for j8 in cutlass.range_constexpr(D // 8):
                        afrag = cute.make_rmem_tensor_like(
                            hmma.make_fragment_A(hmma.partition_shape_A((16, 8))), Int32
                        )
                        for slot, src in ((0, 0), (1, 2), (2, 1), (3, 3)):
                            afrag[slot] = _tf32(res[4 * j8 + src])
                        kcol = 8 * j8 + 2 * c
                        for nt in cutlass.range_constexpr(kt_tiles):
                            tok = 8 * nt + g
                            b_addr = (
                                kn_buf
                                + ((kcol // _BOX) * Tp + tok) * 128
                                + ((((kcol % _BOX) // 4) ^ (tok % 8)) * 16)
                                + (kcol % 4) * 4
                            )
                            bfrag = cute.make_rmem_tensor_like(
                                hmma.make_fragment_B(hmma.partition_shape_B((8, 8))),
                                Int32,
                            )
                            bfrag[0] = _lds_u32(b_addr)
                            bfrag[1] = _lds_u32(b_addr + 4)
                            acc_i = 2 * nt + j8 % 2
                            cute.gemm(hmma, r_acc[acc_i], afrag, bfrag, r_acc[acc_i])
                    # Columns of h0 K^T scaled to normalized keys.
                    for kk in cutlass.range_constexpr(kt_tiles):
                        for hs in cutlass.range_constexpr(2):
                            inv_t = s_inv_b[(local % 2, 8 * kk + 2 * c + hs)]
                            for rh in cutlass.range_constexpr(2):
                                acc_r[4 * kk + hs + 2 * rh] = (
                                    r_acc[2 * kk][hs + 2 * rh]
                                    + r_acc[2 * kk + 1][hs + 2 * rh]
                                ) * inv_t

                    # K^T for the rank update, transposed from s_kn: thread
                    # (k, chunk q) gathers the tokens of permuted positions
                    # 4q..4q+3 down one column (a warp reads 32 columns of one row:
                    # conflict free) and writes one 16-byte chunk of row k.
                    for q_round in cutlass.range_constexpr(Tp // 8):
                        task = tidx + q_round * 128
                        kcol = task % D
                        q = task // D
                        words = []
                        for e in cutlass.range_constexpr(4):
                            tok = (q // 2) * 8 + 2 * e + q % 2
                            words.append(
                                _lds_u32(
                                    kn_buf
                                    + ((kcol // _BOX) * Tp + tok) * 128
                                    + ((((kcol % _BOX) // 4) ^ (tok % 8)) * 16)
                                    + (kcol % 4) * 4
                                )
                            )
                        U.sts_v4(
                            _swizzle(knt_buf + kcol * kt_row + q * 16, kt_mask),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )

                _mbar_wait(s_ready.iterator + buf, (local // NB) % 2)

                scale0 = cutlass.Float32(0.0)
                scale1 = cutlass.Float32(0.0)
                if live:
                    # R, permuted so A slot c holds token 2c and slot c + 4 token
                    # 2c + 1 (where the accumulator fragment keeps them); tokens
                    # t >= a are zeroed.
                    r_frags = []
                    for kk in cutlass.range_constexpr(kt_tiles):
                        parts = [
                            cute.make_rmem_tensor_like(
                                hmma.make_fragment_A(hmma.partition_shape_A((16, 8))),
                                Int32,
                            )
                            for _ in range(2)
                        ]
                        for hs in cutlass.range_constexpr(2):
                            j = 8 * kk + 2 * c + hs
                            beta_j = s_beta[(buf, j)]
                            gam_j = s_gam[(buf, j)]
                            for rh in cutlass.range_constexpr(2):
                                rv = beta_j * (
                                    v_r[(kk * 2 + hs) * 2 + rh]
                                    - gam_j * acc_r[4 * kk + hs + 2 * rh]
                                )
                                if j >= n_commit:
                                    rv = cutlass.Float32(0.0)
                                hi = _tf32_hi(rv)
                                parts[0][rh + 2 * hs] = hi
                                parts[1][rh + 2 * hs] = _tf32(rv - _f32(hi))
                        r_frags.append(parts)

                    # W = R P (and R P_rest with a snapshot) from hi/lo splits
                    # (hi hi + lo hi + hi lo), scaled by the inverse key norms and
                    # split again as rank-update A operands: the C fragment
                    # (g, 2c), (g, 2c+1), (g+8, 2c), (g+8, 2c+1) of token tile nt
                    # is A slot (g, c), (g, c+4), (g+8, c), (g+8, c+4) of k-step nt.
                    for which in cutlass.range_constexpr(2 if HAS_TRACK else 1):
                        p_mat = [s_p, s_pt][which]
                        for nt in cutlass.range_constexpr(kt_tiles):
                            w_acc = hmma.make_fragment_C(
                                hmma.partition_shape_C((16, 8))
                            )
                            w_acc.fill(0.0)
                            for kk in cutlass.range_constexpr(nt + 1):
                                pfrags = []
                                for part in cutlass.range_constexpr(2):
                                    pfrag = cute.make_rmem_tensor_like(
                                        hmma.make_fragment_B(
                                            hmma.partition_shape_B((8, 8))
                                        ),
                                        Int32,
                                    )
                                    for e in cutlass.range_constexpr(2):
                                        pfrag[e] = p_mat[
                                            (buf, part, 8 * nt + g, 8 * kk + 2 * c + e)
                                        ]
                                    pfrags.append(pfrag)
                                for ra, pb in ((0, 0), (1, 0), (0, 1)):
                                    cute.gemm(
                                        hmma, w_acc, r_frags[kk][ra], pfrags[pb], w_acc
                                    )
                            for e, src in ((0, 0), (1, 2), (2, 1), (3, 3)):
                                inv_t = s_inv_b[(local % 2, 8 * nt + 2 * c + src % 2)]
                                wv = w_acc[src] * inv_t
                                hi = _tf32_hi(wv)
                                w_r[(which, nt, 0, e)] = hi
                                w_r[(which, nt, 1, e)] = _tf32(wv - _f32(hi))
                    scale0 = s_scale[(buf, 0)]
                    scale1 = s_scale[(buf, 1)]
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_consumed.iterator + buf)

                if live:
                    # Publish K^T to wgmma.
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    # First result: scale0 h0 + W K, with h0 in FP32.
                    vals = [res[i] * scale0 for i in range(D // 2)]
                    _ptx("wgmma.fence.sync.aligned;")
                    for part in cutlass.range_constexpr(2):
                        for kk in cutlass.range_constexpr(kt_tiles):
                            vals = _wgmma_tf32_rs(
                                vals,
                                [w_r[(0, kk, part, e)] for e in range(4)],
                                _kt_desc(knt_buf + kk * 32, kt_row),
                            )
                    vals = _wgmma_wait(vals)
                    for i in cutlass.range_constexpr(D // 2):
                        res[i] = vals[i]
                    dst = flat
                    if cutlass.const_expr(HAS_TRACK):
                        if tflat >= 0:
                            dst = tflat
                    _stage_and_store(
                        res,
                        out_base,
                        row_base + g,
                        c,
                        is_issuer,
                        tma_atom_st,
                        s_out,
                        tma_rows,
                        dst * HV + hv,
                    )
                    if cutlass.const_expr(HAS_TRACK):
                        if tflat >= 0:
                            # The final state continues from the snapshot.
                            vals2 = [res[i] * scale1 for i in range(D // 2)]
                            _ptx("wgmma.fence.sync.aligned;")
                            for part in cutlass.range_constexpr(2):
                                for kk in cutlass.range_constexpr(kt_tiles):
                                    vals2 = _wgmma_tf32_rs(
                                        vals2,
                                        [w_r[(1, kk, part, e)] for e in range(4)],
                                        _kt_desc(knt_buf + kk * 32, kt_row),
                                    )
                            vals2 = _wgmma_wait(vals2)
                            for i in cutlass.range_constexpr(D // 2):
                                res[i] = vals2[i]
                            _stage_and_store(
                                res,
                                out_base,
                                row_base + g,
                                c,
                                is_issuer,
                                tma_atom_st,
                                s_out,
                                tma_rows,
                                flat * HV + hv,
                            )
                local = local + 1
        if is_issuer:
            cute.arch.cp_async_bulk_wait_group(0)


@cute.jit
def _launch_commit_sm90(
    state_rows: cute.Tensor,
    rawv: cute.Tensor,
    rawk: cute.Tensor,
    log_g: cute.Tensor,
    beta: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    track_indices: cute.Tensor,
    track_steps: cute.Tensor,
    num_items: Int32,
    num_slots: Int32,
    batch: Int32,
    grid_ctas: Int32,
    work: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int],
    NB: cutlass.Constexpr[int],
    CTAS: cutlass.Constexpr[int],
    NULL_BLOCK_ID: cutlass.Constexpr[int],
    HAS_TRACK: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    load = cpasync.CopyBulkTensorTileG2SOp()
    # Each checkpoint moves as one box in both directions: viewed as 128-byte
    # rows (swizzled in shared memory), front to back.
    rows_box = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3),
        0,
        cute.make_layout((2 * _D, _BOX), stride=(_BOX, 1)),
    )
    tma = cpasync.make_tiled_tma_atom(
        load, state_rows, rows_box, cta_tiler=(2 * _D, _BOX)
    )
    tma_st = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        state_rows,
        rows_box,
        cta_tiler=(2 * _D, _BOX),
    )
    # Raw K / V windows: the T rows (128 bytes each, swizzled) of one head.
    row_box = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3),
        0,
        cute.make_layout((1, T, _D), stride=(0, _D, 1)),
    )
    tma_k = cpasync.make_tiled_tma_atom(load, rawk, row_box, cta_tiler=(1, T, _D))
    tma_v = cpasync.make_tiled_tma_atom(load, rawv, row_box, cta_tiler=(1, T, _D))
    gdn_replayssm_commit_kernel_sm90(
        tma.atom,
        tma_st.atom,
        tma_st.tma_tensor,
        tma_k.atom,
        tma_k.tma_tensor,
        tma_v.atom,
        tma_v.tma_tensor,
        log_g,
        beta,
        state_indices,
        accept_lens,
        track_indices,
        track_steps,
        num_items,
        num_slots,
        batch,
        work,
        T,
        H,
        HV,
        S,
        NB,
        NULL_BLOCK_ID,
        HAS_TRACK,
    ).launch(
        grid=(grid_ctas, 1, 1),
        block=(_THREADS, 1, 1),
        max_number_threads=(_THREADS, 1, 1),
        min_blocks_per_mp=CTAS,
        stream=stream,
    )


_CACHE: dict = {}
# Item counters of the dynamic schedule, one pair per (device, stream); see
# the SM90 verify kernel.
_WORK: dict = {}


def _leading_dynamic(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tuple(range(tensor.dim())),
        divisibility=1,
    )


def run_gdn_replayssm_commit_sm90(
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    track_state_indices: torch.Tensor | None,
    track_steps: torch.Tensor | None,
    null_block_id: int,
) -> None:
    """Commit accepted prefixes of normalized ReplaySSM windows on SM90.

    The caller (``commit_gdn_replayssm_fold_all_layers``) validates the
    contract: K = V = 64, T <= 16, shapes, dtypes, contiguity and alignment.
    """
    target = gdn_device_target(checkpoint_state.device)
    num_layers, num_slots, HV = checkpoint_state.shape[:3]
    T = rawk_cache.shape[3]
    H = rawk_cache.shape[2]
    B = state_indices.shape[0]
    has_track = track_state_indices is not None
    num_items = num_layers * B * HV
    stages = _STAGES
    ctas, nbuf = _pipeline_config(T, has_track)
    num_sms = torch.cuda.get_device_properties(
        checkpoint_state.device
    ).multi_processor_count
    grid_ctas = min(num_items, ctas * num_sms)

    state_rows = checkpoint_state.view(-1, _BOX)
    rawv = rawv_cache.view(-1, T, _D)
    rawk = rawk_cache.view(-1, T, _D)
    log_g = g_cache.view(-1)
    beta = beta_cache.view(-1)
    track_indices = track_state_indices if has_track else state_indices
    steps = track_steps if has_track else accept_lens

    key = (
        T,
        H,
        HV,
        stages,
        nbuf,
        ctas,
        int(null_block_id),
        has_track,
        target.compile_key,
    )
    cache = _CACHE.setdefault(key, {})
    torch_stream = torch.cuda.current_stream(device=checkpoint_state.device)
    work = _WORK.get((checkpoint_state.device, torch_stream.cuda_stream))
    if work is None:
        work = torch.zeros(2, dtype=torch.int32, device=checkpoint_state.device)
        _WORK[(checkpoint_state.device, torch_stream.cuda_stream)] = work
    stream = cuda.CUstream(torch_stream.cuda_stream)
    if "compiled" not in cache:
        compile_options = gdn_compile_options(
            checkpoint_state.device,
            cute.EnableTVMFFI(True),
            cute.GenerateLineInfo(True),
        )
        cache["compiled"] = build_and_load_cute_dsl_kernel(
            "gdn_replayssm_commit_sm90",
            make_kernel_name("commit_sm90", target.arch, *key[:8]),
            lambda: cute.compile[compile_options](
                _launch_commit_sm90,
                _leading_dynamic(state_rows),
                _leading_dynamic(rawv),
                _leading_dynamic(rawk),
                _leading_dynamic(log_g),
                _leading_dynamic(beta),
                _leading_dynamic(state_indices),
                _leading_dynamic(accept_lens),
                _leading_dynamic(track_indices),
                _leading_dynamic(steps),
                Int32(num_items),
                Int32(num_slots),
                Int32(B),
                Int32(grid_ctas),
                from_dlpack(work),
                T=T,
                H=H,
                HV=HV,
                S=stages,
                NB=nbuf,
                CTAS=ctas,
                NULL_BLOCK_ID=int(null_block_id),
                HAS_TRACK=has_track,
                stream=stream,
            ),
            extra_key_files=(__file__,),
        )
    cache["compiled"](
        state_rows,
        rawv,
        rawk,
        log_g,
        beta,
        state_indices,
        accept_lens,
        track_indices,
        steps,
        num_items,
        num_slots,
        B,
        grid_ctas,
        work,
        stream,
    )
