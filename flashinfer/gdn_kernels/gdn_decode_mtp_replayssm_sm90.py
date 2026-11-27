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

"""SM90 ReplaySSM verify: persistent, TMA-pipelined, TF32 wgmma.

Work item: one batch row and the ``HPC`` value heads that share one Q/K head.
Verify reads each frozen FP32 ``D x D`` checkpoint exactly once, so the kernel
is a streaming problem and is organized to keep that stream busy:

* Persistent CTAs (one or two per SM, see ``_pipeline_config``) with three
  warp roles.  Items are scheduled dynamically from a per-stream counter
  (see the producer) so the CTAs finish together.  A producer warp streams items into an ``S``-stage shared-memory
  ring with TMA: the raw Q/K/V rows and the 16-byte gate chunks holding the
  item's heads first, on their own barrier, then the checkpoint as 64 x 32
  FP32 boxes in the 128-byte swizzle (the K-major layout ``wgmma`` reads
  directly).  Two auxiliary warps take
  alternate items and prepare everything that depends only on Q/K and the
  gates, up to ``NB`` items ahead.  One warpgroup per value head (the base
  warps) does the checkpoint contractions.  Roles hand off through mbarriers
  only.  The raw windows and the output leave through TMA stores.
* The recurrence needs no per-token state update.  With ``gamma_t = exp(L_t)``
  the cumulative decay, ``BK = h0 K^T`` and ``BQ = h0 Q^T`` (V x T), the
  single-chunk delta rule gives

      O = BQ diag(gamma) + R M,   R = (V - BK diag(gamma)) diag(beta)
      M = (I + Lk^T diag(beta))^-1 P^T

  with ``Lk[t, j] = gamma_t / gamma_j k_j.k_t`` (j < t) and
  ``P[t, j] = gamma_t / gamma_j k_j.q_t`` (j <= t).  The decay ratio factors
  out of the triangular solve, ``M[j, t] = gamma_t / gamma_j M'[j, t]`` with

      M'[j, t] = (k_j.q_t) - sum_{j < i <= t} (k_j.k_i) beta_i M'[i, t],

  which depends only on the Gram matrix and beta, so it never divides by a
  (possibly tiny) cumulative decay.
* ``h0 [K; Q]^T`` is one ``wgmma`` m64nNk8 chain per value head reading the
  checkpoint and the normalized operand straight from shared memory; ``R M``
  uses ``mma.sync`` on ``R`` formed in registers from the wgmma accumulators,
  whose per-warp layout matches the ``mma.sync`` C fragment.  The base warps
  normalize Q/K themselves, so they never wait for the aux warps before their
  checkpoint contraction.
* The aux warps work from the raw rows so they can run ahead: they form the
  raw Gram matrix ``G`` of ``[K; Q]`` with BF16 ``mma`` (exact products), take
  the inverse norms ``n`` from its diagonal and solve on ``G`` directly, using
  ``M'[j, t] = n_j Y[j, t]`` with

      Y[j, t] = nq_t G[j, q_t] - sum_{j < i <= t} G[j, i] beta_i n_i^2 Y[i, t].

``wgmma`` reads the FP32 checkpoint as TF32 (low mantissa bits ignored), as
the SM100 tcgen05 path does.  Operands written by the kernel are rounded to
nearest TF32 by adding half a TF32 ulp to the FP32 bit pattern; SM90 has no
native ``cvt.rna.tf32.f32``.
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
import cuda.bindings.driver as cuda


_ACC = cutlass.Float32
_D = 64
_BOX = 32  # FP32 columns per 128-byte swizzled row
# Dynamic shared memory per SM available to the kernel's CTAs.
_SMEM_PER_SM = 227 * 1024
_TF32_HALF_ULP = 0x1000
# Row stride (words) of M^T.  The m16n8k8 B-fragment LDS.64 of lane (g, c)
# reads words 24 g + 2 c: conflict free in each 16-lane phase for Tp <= 16.
_M_ROW = 24


def _tpad(T: int) -> int:
    return (T + 7) // 8 * 8


def _ab_block(T: int) -> int:
    # TMA destinations must be 128-byte aligned: pad each a/b block.
    return (T * 8 + 63) // 64 * 64


def _tx_bytes(T: int, hpc: int) -> int:
    return hpc * _D * _D * 4 + 2 * T * _D * 2 + T * hpc * _D * 2 + 2 * T * 16


def _stage_bytes(T: int, hpc: int) -> int:
    # Q/K blocks hold Tp swizzled rows (the box loads T of them).
    return _tx_bytes(T, hpc) + (2 + hpc) * (_tpad(T) - T) * _D * 2 + 2 * 128 + 1024


def _work_bytes(T: int, hpc: int, HV: int, nbuf: int) -> int:
    tp = _tpad(T)
    per_buf = hpc * tp * _M_ROW * 4 + 2 * hpc * 16 * 4
    per_aux = 2 * tp * (2 * tp + 4) * 4 + 32 * 4 + 2 * hpc * 16 * 4
    opnd = 2 * 2 * tp * _D * 4
    out = tp * hpc * _D * 2 + 1024
    return nbuf * per_buf + 2 * per_aux + opnd + out + 2 * HV * 4 + 3072


def _pipeline_config(T: int, hpc: int, HV: int) -> tuple:
    """Pick (CTAs per SM, checkpoint stages, aux->base buffers).

    Two CTAs per SM with two stages each beat one CTA with four stages
    wherever they fit (everything but two heads per item at T > 8): the two
    independent consumer chains overlap each other's per-item latency, and two
    stages already saturate the checkpoint stream.  Otherwise one CTA takes a
    four-stage ring.

    The stage count is even: the two aux warps take alternate items, so the
    previous use of a stage always belongs to the same aux warp and a parity
    wait can never alias a phase that is still in flight.  (Aux->base buffers
    are released by the in-order base warps, so any count works.)  More
    buffers let the aux warps run further ahead of the base warps.
    """
    for ctas, stages, nbuf in ((2, 2, 4), (1, 4, 4)):
        budget = _SMEM_PER_SM // ctas - 1024  # 1 KiB reserved per CTA
        if stages * _stage_bytes(T, hpc) + _work_bytes(T, hpc, HV, nbuf) <= budget:
            return ctas, stages, nbuf
    raise ValueError(
        f"SM90 ReplaySSM verify: T={T}, HV={HV} does not fit in shared memory"
    )


@dsl_user_op
def _bits(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            _T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32(w: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            _T.f32(),
            [Int32(w).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mbar_wait(mbar, phase, *, loc=None, ip=None) -> None:
    """Spin on ``mbarrier.try_wait.parity`` without a sleep back-off.

    The handoffs here are short and latency-critical; a nanosleep back-off
    after a failed try_wait can oversleep them.
    """
    llvm.inline_asm(
        None,
        [
            Int32(mbar.toint()).ir_value(loc=loc, ip=ip),
            Int32(phase).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tWAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n\t"
        "@!p bra WAIT;\n\t}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _evict_first_policy(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            _T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ptx(text: str, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [], text, "", has_side_effects=True, is_align_stack=False, loc=loc, ip=ip
    )


@dsl_user_op
def _wgmma_tf32(acc, a_desc, b_desc, accumulate, *, loc=None, ip=None):
    """``wgmma.mma_async`` m64nNk8 TF32 with both operands in shared memory.

    ``acc`` holds the N / 2 FP32 accumulators of this thread; they are tied
    in and out so the chain of k-steps keeps one register set.
    """
    nreg = len(acc)
    regs = ",".join(f"${i}" for i in range(nreg))
    text = (
        "{\n\t.reg .pred p;\n\t"
        f"setp.ne.b32 p, ${nreg + 2}, 0;\n\t"
        f"wgmma.mma_async.sync.aligned.m64n{2 * nreg}k8.f32.tf32.tf32 "
        f"{{{regs}}}, ${nreg}, ${nreg + 1}, p, 1, 1;\n\t}}"
    )
    constraints = (
        ",".join(["=f"] * nreg) + ",l,l,r," + ",".join(str(i) for i in range(nreg))
    )
    operands = [
        Int64(a_desc).ir_value(loc=loc, ip=ip),
        Int64(b_desc).ir_value(loc=loc, ip=ip),
        Int32(accumulate).ir_value(loc=loc, ip=ip),
    ] + [Float32(x).ir_value(loc=loc, ip=ip) for x in acc]
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
def _wgmma_wait(acc, *, loc=None, ip=None):
    """Commit and wait for the wgmma group; the accumulators pass through so
    no read of them can be scheduled before the wait."""
    nreg = len(acc)
    text = "wgmma.commit_group.sync.aligned;\n\twgmma.wait_group.sync.aligned 0;"
    constraints = ",".join(["=f"] * nreg) + "," + ",".join(str(i) for i in range(nreg))
    st = llvm.StructType.get_literal([_T.f32()] * nreg)
    r = llvm.inline_asm(
        st,
        [Float32(x).ir_value(loc=loc, ip=ip) for x in acc],
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


@cute.jit
def _sw128_desc(addr: Int32) -> Int64:
    """wgmma descriptor of a K-major operand in the 128-byte swizzle: 8-row
    atoms of 128-byte rows, 1024 bytes apart."""
    return (
        (Int64(addr) >> 4)
        | (Int64(1) << 16)
        | (Int64(1024 >> 4) << 32)
        | (Int64(1) << 62)
    )


@dsl_user_op
def _lds_u32(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            _T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _qk_row_addr(qk_stage: Int32, r: Int32, Tp: cutlass.Constexpr[int]) -> Int32:
    """Byte address of operand row r: K rows at [0, Tp), Q rows at [Tp, 2Tp)."""
    kind = Int32(1)
    rr = r
    if r >= Tp:
        kind = Int32(0)
        rr = r - Tp
    return qk_stage + kind * (Tp * _D * 2) + rr * (_D * 2)


@cute.jit
def _tf32(x: Float32) -> Int32:
    return _bits(x) + _TF32_HALF_ULP


@cute.jit
def _tma_load(atom, gmem, smem, mbar, policy):
    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )
    if cutlass.const_expr(policy is None):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar)
    else:
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar, cache_policy=policy)


@cute.kernel
def gdn_verify_kernel_mtp_replayssm_sm90(
    tma_atom: cute.CopyAtom,
    tma_state: cute.Tensor,
    tma_atom_q: cute.CopyAtom,
    tma_q: cute.Tensor,
    tma_atom_k: cute.CopyAtom,
    tma_k: cute.Tensor,
    tma_atom_v: cute.CopyAtom,
    tma_v: cute.Tensor,
    tma_atom_a: cute.CopyAtom,
    tma_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_b: cute.Tensor,
    tma_atom_rawv: cute.CopyAtom,
    tma_rawv: cute.Tensor,
    tma_atom_rawk: cute.CopyAtom,
    tma_rawk: cute.Tensor,
    tma_atom_out: cute.CopyAtom,
    tma_out: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    state_indices: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    num_items: Int32,
    work: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HPC: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int],
    NB: cutlass.Constexpr[int],
):
    D = _D
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    base_warps = HPC * D // 16  # one warpgroup per value head
    producer_warp = base_warps + 2
    all_threads = (base_warps + 3) * 32
    groups_per_q = HV // H // HPC
    items_per_n = H * groups_per_q
    Tp = _tpad(T)
    NP = 2 * Tp
    n_tiles = NP // 8
    kt_tiles = Tp // 8
    halves = D // _BOX
    ab_block = _ab_block(T)

    smem = utils.SmemAllocator()
    s_state = smem.allocate_tensor(
        _ACC,
        cute.make_layout(
            (S, HPC, D, (_BOX, halves)),
            stride=(HPC * D * D, D * D, _BOX, (1, D * _BOX)),
        ),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    # Base-private normalized operand (double-buffered by item) as TF32 bits,
    # K-major in the 128-byte swizzle like the checkpoint: [buf, half, row,
    # 32 words], K rows at [0, T), Q rows at [Tp, Tp + T), zero elsewhere.
    s_opnd = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout((2 * halves * NP * _BOX,)), 1024
    )
    # Raw Q (kind 0) and K (kind 1) rows, 128-byte swizzled; rows [T, Tp) are
    # never loaded and are masked on read.
    s_qk = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((S, 2, Tp, D), stride=(2 * Tp * D, Tp * D, D, 1)),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    # V rows per head, [stage, head, t, D] (the raw V window layout), in the
    # 128-byte swizzle: the R build's lanes c read tokens 2c, 2c + 1 of one
    # V row, which the swizzle spreads over distinct banks.
    s_v = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((S, HPC, Tp, D), stride=(HPC * Tp * D, Tp * D, D, 1)),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    # The 16-byte a/b chunk of each token that holds this item's heads.
    s_ab = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((S, 2, T, 8), stride=(2 * ab_block, ab_block, 8, 1)),
        128,
    )
    # Per stage: s_rows signals the Q/K/V rows and gate chunks, s_full the
    # checkpoint, s_empty that every consumer is done with the stage.
    s_rows = smem.allocate_tensor(cutlass.Int64, cute.make_layout((S,)), 8)
    s_full = smem.allocate_tensor(cutlass.Int64, cute.make_layout((S,)), 8)
    s_empty = smem.allocate_tensor(cutlass.Int64, cute.make_layout((S,)), 8)
    s_ready = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NB,)), 8)
    s_consumed = smem.allocate_tensor(cutlass.Int64, cute.make_layout((NB,)), 8)
    # M^T per head as TF32 bits: s_m[buf, head, t, j] = M[j, t].
    s_m = smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout(
            (NB, HPC, Tp, _M_ROW),
            stride=(HPC * Tp * _M_ROW, Tp * _M_ROW, _M_ROW, 1),
        ),
        16,
    )
    buf_layout = cute.make_layout((NB, HPC, 16), stride=(HPC * 16, 16, 1))
    s_gam = smem.allocate_tensor(_ACC, buf_layout, 16)
    s_beta = smem.allocate_tensor(_ACC, buf_layout, 16)
    # Aux-private scratch: cumulative log decay, beta n^2, inverse norms of
    # [K; Q] (Q including scale) and the raw Gram matrix.
    aux_layout = cute.make_layout((2, HPC, 16), stride=(HPC * 16, 16, 1))
    s_lam = smem.allocate_tensor(_ACC, aux_layout, 16)
    s_betap = smem.allocate_tensor(_ACC, aux_layout, 16)
    s_inv = smem.allocate_tensor(_ACC, cute.make_layout((2, 32), stride=(32, 1)), 16)
    s_g = smem.allocate_tensor(
        _ACC,
        cute.make_layout((2, NP, NP + 4), stride=(NP * (NP + 4), NP + 4, 1)),
        16,
    )
    # Output staging [t, head, D] in the 128-byte swizzle, stored with TMA.
    s_out = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((Tp, HPC, D), stride=(HPC * D, D, 1)),
        1024,
        swizzle=cute.make_swizzle(3, 4, 3),
    )
    # Checkpoint slot of the item in each stage (negative: padding row).
    s_slot = smem.allocate_tensor(cutlass.Int32, cute.make_layout((S,)), 16)
    # Item of each stage (negative: no more items).
    s_item = smem.allocate_tensor(cutlass.Int32, cute.make_layout((S,)), 16)
    s_neg_a = smem.allocate_tensor(_ACC, cute.make_layout((HV,)), 16)
    s_dtb = smem.allocate_tensor(_ACC, cute.make_layout((HV,)), 16)

    if tidx == 0:
        for s in cutlass.range_constexpr(S):
            cute.arch.mbarrier_init(s_rows.iterator + s, 1)
            cute.arch.mbarrier_init(s_full.iterator + s, 1)
            cute.arch.mbarrier_init(s_empty.iterator + s, base_warps + 1)
        for s in cutlass.range_constexpr(NB):
            cute.arch.mbarrier_init(s_ready.iterator + s, 1)
            cute.arch.mbarrier_init(s_consumed.iterator + s, base_warps)
    cute.arch.mbarrier_init_fence()
    # The producer reads its first slot index before the CTA barrier, so the
    # load latency overlaps the prologue instead of delaying the first TMA.
    first_slot = Int32(0)
    if warp == producer_warp:
        first_slot = state_indices[bid // items_per_n]
    opnd_words = 2 * halves * NP * _BOX
    for i in cutlass.range_constexpr((opnd_words + all_threads - 1) // all_threads):
        linear = tidx + i * all_threads
        if linear < opnd_words:
            s_opnd[linear] = 0
    # The operand is read by wgmma through the async proxy.
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()

    qk_addr = s_qk.iterator.toint()
    qk_stage_bytes = 2 * Tp * D * 2
    opnd_addr = s_opnd.iterator.toint()
    state_addr = s_state.iterator.toint()

    # ------------------------------------------------------------------ producer
    if warp == producer_warp:
        state_bytes = HPC * D * D * 4
        row_bytes = _tx_bytes(T, HPC) - state_bytes
        # Each checkpoint byte is read exactly once.
        evict_first = _evict_first_policy()
        local = Int32(0)
        # Dynamic schedule: each CTA takes items bid and bid + grid, then the
        # next unclaimed items from a global counter, so CTAs the memory system
        # serves faster take more items and all finish together.  The static
        # start keeps small launches free of counter traffic.  The claim for
        # the next item and its slot index are in flight while this item is
        # issued; the producer passes the item to the other roles in s_item.
        item = bid
        next_slot = first_slot
        while item < num_items:
            stage = local % S
            claim = Int32(0)
            if lane == 0 and local > 0:
                claim = cute.arch.atomic_add(work.iterator.llvm_ptr, Int32(1))
            cache_idx = next_slot
            if local >= S:
                _mbar_wait(s_empty.iterator + stage, ((local // S) - 1) % 2)
            i_n = item // items_per_n
            i_group = item % items_per_n
            i_h = i_group // groups_per_q
            hv0 = i_group * HPC
            slot = cache_idx
            if slot < 0:
                slot = Int32(0)
            # The small rows go first on their own barrier: the aux solve and
            # the Q/K normalization start while the checkpoint is in flight,
            # leaving only the checkpoint contraction behind the last load.
            rows = s_rows.iterator + stage
            full = s_full.iterator + stage
            if lane == 0:
                s_slot[stage] = cache_idx
                s_item[stage] = item
                cute.arch.mbarrier_arrive_and_expect_tx(rows, row_bytes)
                cute.arch.mbarrier_arrive_and_expect_tx(full, state_bytes)
            cute.arch.sync_warp()
            _tma_load(
                tma_atom_q,
                cute.local_tile(tma_q, (1, T, 1, D), (i_n, 0, i_h, 0))[
                    (0, None, 0, None)
                ],
                cute.local_tile(s_qk[(stage, 0, None, None)], (T, D), (0, 0)),
                rows,
                None,
            )
            _tma_load(
                tma_atom_k,
                cute.local_tile(tma_k, (1, T, 1, D), (i_n, 0, i_h, 0))[
                    (0, None, 0, None)
                ],
                cute.local_tile(s_qk[(stage, 1, None, None)], (T, D), (0, 0)),
                rows,
                None,
            )
            for head in cutlass.range_constexpr(HPC):
                _tma_load(
                    tma_atom_v,
                    cute.local_tile(tma_v, (1, T, 1, D), (i_n, 0, hv0 + head, 0))[
                        (0, None, 0, None)
                    ],
                    cute.local_tile(s_v[(stage, head, None, None)], (T, D), (0, 0)),
                    rows,
                    None,
                )
            _tma_load(
                tma_atom_a,
                cute.local_tile(tma_a, (1, T, 8), (i_n, 0, hv0 // 8))[(0, None, None)],
                s_ab[(stage, 0, None, None)],
                rows,
                None,
            )
            _tma_load(
                tma_atom_b,
                cute.local_tile(tma_b, (1, T, 8), (i_n, 0, hv0 // 8))[(0, None, None)],
                s_ab[(stage, 1, None, None)],
                rows,
                None,
            )
            for head in cutlass.range_constexpr(HPC):
                for half in cutlass.range_constexpr(halves):
                    _tma_load(
                        tma_atom,
                        tma_state[(slot * HV + hv0 + head, None, (None, half))],
                        s_state[(stage, head, None, (None, half))],
                        full,
                        evict_first,
                    )
            local = local + 1
            item = bid + grid
            if local > 1:
                item = 2 * grid + cute.arch.shuffle_sync(claim, 0)
            if item < num_items:
                next_slot = state_indices[item // items_per_n]
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
                s_item[stage] = Int32(-1)
                cute.arch.mbarrier_arrive(s_rows.iterator + stage)
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
        # Only the aux warps use the gate constants; loading them here keeps
        # their global-load latency out of the CTA prologue.
        for i in cutlass.range_constexpr((HV + 63) // 64):
            h_idx = aux * 32 + lane + i * 64
            if h_idx < HV:
                s_neg_a[h_idx] = -cute.exp(cutlass.Float32(A_log[h_idx]), fastmath=True)
                s_dtb[h_idx] = cutlass.Float32(dt_bias[h_idx])
        cute.arch.barrier(barrier_id=2, number_of_threads=64)
        local = aux
        go = Int32(1)
        while go > 0:
            stage = local % S
            buf = local % NB
            _mbar_wait(s_rows.iterator + stage, (local // S) % 2)
            item = s_item[stage]
            if item < 0:
                go = Int32(0)
            else:
                hv0 = (item % items_per_n) * HPC
                cache_idx = s_slot[stage]
                valid = cache_idx >= 0
                if local >= NB:
                    _mbar_wait(s_consumed.iterator + buf, ((local // NB) - 1) % 2)
                # Gates: lane = head * 16 + t.
                head = lane // 16
                t = lane % 16
                log_decay = cutlass.Float32(0.0)
                beta_value = cutlass.Float32(0.0)
                if head < HPC and t < T:
                    hv = hv0 + head
                    x = cutlass.Float32(s_ab[(stage, 0, t, hv % 8)]) + s_dtb[hv]
                    softplus_x = x
                    if x <= 20.0:
                        softplus_x = cute.log(
                            cutlass.Float32(1.0) + cute.exp(x, fastmath=True),
                            fastmath=True,
                        )
                    log_decay = s_neg_a[hv] * softplus_x
                    beta_value = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0)
                        + cute.exp(
                            -cutlass.Float32(s_ab[(stage, 1, t, hv % 8)]), fastmath=True
                        )
                    )
                    if valid:
                        replayssm_g[(cache_idx, hv, t)] = log_decay
                        replayssm_beta[(cache_idx, hv, t)] = beta_value
                lam = log_decay
                for offset in [1, 2, 4, 8]:
                    other = cute.arch.shuffle_sync_up(
                        lam, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    if t >= offset:
                        lam = lam + other

                # Raw Gram [K; Q] [K; Q]^T with BF16 m16n8k16 (exact products),
                # skipping an all-K m-tile's Q columns: the solve reads K.K and
                # Q.K rows (contiguous in j) and the norms the diagonal.  Under the 128-byte swizzle
                # lane (g, c) reads word c of chunk (2 ks + m) ^ (row % 8), which
                # is conflict free; rows [T, Tp) of each block read as zero.
                qk_stage = qk_addr + stage * qk_stage_bytes
                for mt in cutlass.range_constexpr(NP // 16):
                    a_rows = []
                    a_ok = []
                    for rh in cutlass.range_constexpr(2):
                        r = 16 * mt + g + 8 * rh
                        a_rows.append(_qk_row_addr(qk_stage, r, Tp))
                        a_ok.append((r % Tp) < T)
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
                    for nt in cutlass.range_constexpr(n_tiles):
                        if cutlass.const_expr(
                            not (16 * mt + 16 <= Tp and 8 * nt >= Tp)
                        ):
                            b_r = 8 * nt + g
                            b_row = _qk_row_addr(qk_stage, b_r, Tp)
                            b_ok = (b_r % Tp) < T
                            gacc = [cutlass.Float32(0.0)] * 4
                            for ks in cutlass.range_constexpr(D // 16):
                                b_w = []
                                for m in cutlass.range_constexpr(2):
                                    w = _lds_u32(
                                        b_row + (((2 * ks + m) ^ g) * 16) + c * 4
                                    )
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
                                    s_g[
                                        (aux, 16 * mt + g + 8 * rh, 8 * nt + 2 * c + hs)
                                    ] = gacc[hs + 2 * rh]
                cute.arch.sync_warp()
                # The stage's Q/K/gate rows are no longer needed by this warp.
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_empty.iterator + stage)
                if lane < NP:
                    inv = cute.rsqrt(s_g[(aux, lane, lane)] + 1e-6, fastmath=True)
                    if lane >= Tp:
                        inv = inv * scale
                    s_inv[(aux, lane)] = inv
                cute.arch.sync_warp()
                if head < HPC and t < Tp:
                    inv_k = s_inv[(aux, t)]
                    s_lam[(aux, head, t)] = lam
                    s_betap[(aux, head, t)] = beta_value * inv_k * inv_k
                    s_gam[(buf, head, t)] = cute.exp(lam, fastmath=True)
                    s_beta[(buf, head, t)] = beta_value
                cute.arch.sync_warp()

                # Column t of Y per lane, right-looking back substitution on the
                # raw Gram matrix (see the module docstring); every Gram row and
                # per-token factor is read with 16-byte loads.
                if head < HPC and t < Tp:
                    g_aux = s_g.iterator.toint() + aux * (NP * (NP + 4) * 4)
                    tok = aux * (HPC * 16) + head * 16
                    betap = []
                    lams = []
                    invs = []
                    for jq in cutlass.range_constexpr(Tp // 4):
                        betap += U.lds_v4(s_betap.iterator.toint() + (tok + 4 * jq) * 4)
                        lams += U.lds_v4(s_lam.iterator.toint() + (tok + 4 * jq) * 4)
                        invs += U.lds_v4(
                            s_inv.iterator.toint() + (aux * 32 + 4 * jq) * 4
                        )
                    inv_q = s_inv[(aux, Tp + t)]
                    ysol = []
                    for jq in cutlass.range_constexpr(Tp // 4):
                        w4 = U.lds_v4(g_aux + ((Tp + t) * (NP + 4) + 4 * jq) * 4)
                        for e in cutlass.range_constexpr(4):
                            j = 4 * jq + e
                            init = inv_q * _f32(Int32(w4[e]))
                            if j > t or t >= T:
                                init = cutlass.Float32(0.0)
                            ysol.append(init)
                    for ii in cutlass.range_constexpr(Tp):
                        i = Tp - 1 - ii
                        coef = _f32(Int32(betap[i])) * ysol[i]
                        for jq in cutlass.range_constexpr((i + 3) // 4):
                            w4 = U.lds_v4(g_aux + (i * (NP + 4) + 4 * jq) * 4)
                            for e in cutlass.range_constexpr(4):
                                j = 4 * jq + e
                                if cutlass.const_expr(j < i):
                                    ysol[j] = ysol[j] - _f32(Int32(w4[e])) * coef
                    lam_t = s_lam[(aux, head, t)]
                    vals = []
                    for j in cutlass.range_constexpr(Tp):
                        val = (
                            cute.exp(lam_t - _f32(Int32(lams[j])), fastmath=True)
                            * _f32(Int32(invs[j]))
                            * ysol[j]
                        )
                        if j > t:
                            val = cutlass.Float32(0.0)
                        vals.append(_tf32(val))
                    m_row = (
                        s_m.iterator.toint()
                        + (((buf * HPC + head) * Tp + t) * _M_ROW) * 4
                    )
                    for jq in cutlass.range_constexpr(Tp // 4):
                        U.sts_v4(
                            m_row + 16 * jq,
                            vals[4 * jq],
                            vals[4 * jq + 1],
                            vals[4 * jq + 2],
                            vals[4 * jq + 3],
                        )
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_ready.iterator + buf)
                local = local + 2

    # ------------------------------------------------------------------ base
    if warp < base_warps:
        head_w = warp // (D // 16)
        row_base = (warp % (D // 16)) * 16
        local = Int32(0)
        go = Int32(1)
        while go > 0:
            stage = local % S
            buf = local % NB
            _mbar_wait(s_rows.iterator + stage, (local // S) % 2)
            item = s_item[stage]
            if item < 0:
                go = Int32(0)
            else:
                i_n = item // items_per_n
                i_group = item % items_per_n
                i_h = i_group // groups_per_q
                hv0 = i_group * HPC
                cache_idx = s_slot[stage]
                valid = cache_idx >= 0

                # Raw windows go out with TMA straight from the stage.
                if warp == 0 and valid:
                    U.simple_tma_copy(
                        tma_atom_rawv,
                        cute.local_tile(
                            s_v[(stage, None, None, None)], (HPC, T, D), (0, 0, 0)
                        ),
                        cute.local_tile(
                            tma_rawv, (1, HPC, T, D), (cache_idx, hv0 // HPC, 0, 0)
                        )[(0, None, None, None)],
                    )
                    if hv0 % (HV // H) == 0:
                        U.simple_tma_copy(
                            tma_atom_rawk,
                            cute.local_tile(
                                s_qk[(stage, 1, None, None)], (T, D), (0, 0)
                            ),
                            cute.local_tile(
                                tma_rawk, (1, 1, T, D), (cache_idx, i_h, 0, 0)
                            )[(0, 0, None, None)],
                        )
                    cute.arch.cp_async_bulk_commit_group()

                # Q/K L2 normalization into the operand buffer: 8 lanes per row,
                # one 16-byte chunk (8 BF16) each; the raw rows are swizzled
                # (chunk XOR row % 8) and so is the operand (16-byte chunks
                # 2 (sub % 4) + m of half sub // 4, XOR row % 8).
                opnd_buf = opnd_addr + (local % 2) * (halves * NP * 128)
                base_tid = warp * 32 + lane
                for r_round in cutlass.range_constexpr(
                    (2 * T * 8 + base_warps * 32 - 1) // (base_warps * 32)
                ):
                    r = base_tid // 8 + r_round * (base_warps * 4)
                    sub = base_tid % 8
                    kind = Int32(1)
                    t = r
                    dst_row = r
                    mult = cutlass.Float32(1.0)
                    if r >= T:
                        kind = Int32(0)
                        t = r - T
                        dst_row = Tp + r - T
                        mult = cutlass.Float32(scale)
                    row_ok = r < 2 * T
                    if not row_ok:
                        t = Int32(0)
                    w4 = U.lds_v4(
                        qk_addr
                        + stage * qk_stage_bytes
                        + kind * (Tp * D * 2)
                        + t * (D * 2)
                        + ((sub ^ (t % 8)) * 16)
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
                    inv = cute.rsqrt(ss + 1e-6, fastmath=True) * mult
                    if row_ok:
                        row_addr = opnd_buf + ((sub // 4) * NP + dst_row) * 128
                        for m in cutlass.range_constexpr(2):
                            U.sts_v4(
                                row_addr + (((2 * (sub % 4) + m) ^ (dst_row % 8)) * 16),
                                _tf32(xs[4 * m] * inv),
                                _tf32(xs[4 * m + 1] * inv),
                                _tf32(xs[4 * m + 2] * inv),
                                _tf32(xs[4 * m + 3] * inv),
                            )
                # Publish the operand to the async proxy (wgmma).  Warp 0 also waits
                # until the bulk stores have read their shared memory: the previous
                # output store (the staging tile is rewritten below) and this
                # item's raw windows (the stage is released after the wgmma).
                # Overlapping the raw-window stores with the wgmma measured no
                # faster.
                cute.arch.fence_proxy("async.shared", space="cta")
                if warp == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=1, number_of_threads=base_warps * 32)

                # h0 [K; Q]^T: one m64nNk8 wgmma chain per value head (warpgroup).
                _mbar_wait(s_full.iterator + stage, (local // S) % 2)
                a_head = state_addr + (stage * HPC + head_w) * (D * D * 4)
                acc = [cutlass.Float32(0.0)] * (NP // 2)
                _ptx("wgmma.fence.sync.aligned;")
                for ks in cutlass.range_constexpr(D // 8):
                    k_off = (ks // 4) * (D * _BOX * 4) + (ks % 4) * 32
                    b_off = (ks // 4) * (NP * 128) + (ks % 4) * 32
                    acc = _wgmma_tf32(
                        acc,
                        _sw128_desc(a_head + k_off),
                        _sw128_desc(opnd_buf + b_off),
                        Int32(0 if ks == 0 else 1),
                    )

                # V values this lane needs for R.
                v_vals = []
                for kk in cutlass.range_constexpr(kt_tiles):
                    for hs in cutlass.range_constexpr(2):
                        j_safe = 8 * kk + 2 * c + hs
                        if j_safe >= T:
                            j_safe = T - 1
                        for rh in cutlass.range_constexpr(2):
                            v_vals.append(
                                cutlass.Float32(
                                    s_v[(stage, head_w, j_safe, row_base + g + 8 * rh)]
                                )
                            )

                acc = _wgmma_wait(acc)
                # The stage (checkpoint, Q/K/V) is no longer read.
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_empty.iterator + stage)
                _mbar_wait(s_ready.iterator + buf, (local // NB) % 2)

                # O = BQ diag(gamma) + R M.  The R M contraction is permuted so A
                # slot c holds j = 2c and slot c + 4 holds j = 2c + 1, which is
                # where the accumulator fragment of BK keeps them.
                r_frags = []
                for kk in cutlass.range_constexpr(kt_tiles):
                    rfrag = cute.make_rmem_tensor_like(
                        hmma.make_fragment_A(hmma.partition_shape_A((16, 8))), Int32
                    )
                    for hs in cutlass.range_constexpr(2):
                        j = 8 * kk + 2 * c + hs
                        beta_j = s_beta[(buf, head_w, j)]
                        gam_j = s_gam[(buf, head_w, j)]
                        for rh in cutlass.range_constexpr(2):
                            rfrag[rh + 2 * hs] = _tf32(
                                beta_j
                                * (
                                    v_vals[(kk * 2 + hs) * 2 + rh]
                                    - gam_j * acc[4 * kk + hs + 2 * rh]
                                )
                            )
                    r_frags.append(rfrag)
                for nt in cutlass.range_constexpr(kt_tiles):
                    o_acc = hmma.make_fragment_C(hmma.partition_shape_C((16, 8)))
                    for hs in cutlass.range_constexpr(2):
                        gam_t = s_gam[(buf, head_w, 8 * nt + 2 * c + hs)]
                        for rh in cutlass.range_constexpr(2):
                            o_acc[hs + 2 * rh] = (
                                acc[4 * (kt_tiles + nt) + hs + 2 * rh] * gam_t
                            )
                    for kk in cutlass.range_constexpr(nt + 1):
                        mfrag = cute.make_rmem_tensor_like(
                            hmma.make_fragment_B(hmma.partition_shape_B((8, 8))), Int32
                        )
                        mfrag[0] = s_m[(buf, head_w, 8 * nt + g, 8 * kk + 2 * c)]
                        mfrag[1] = s_m[(buf, head_w, 8 * nt + g, 8 * kk + 2 * c + 1)]
                        cute.gemm(hmma, o_acc, r_frags[kk], mfrag, o_acc)
                    for hs in cutlass.range_constexpr(2):
                        for rh in cutlass.range_constexpr(2):
                            s_out[
                                (8 * nt + 2 * c + hs, head_w, row_base + g + 8 * rh)
                            ] = cutlass.BFloat16(o_acc[hs + 2 * rh])
                if lane == 0:
                    cute.arch.mbarrier_arrive(s_consumed.iterator + buf)
                # Output tile out with one TMA store once every base warp wrote it.
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=1, number_of_threads=base_warps * 32)
                if warp == 0 and valid:
                    U.simple_tma_copy(
                        tma_atom_out,
                        cute.local_tile(s_out, (T, HPC, D), (0, 0, 0)),
                        cute.local_tile(
                            tma_out, (1, T, HPC, D), (i_n, 0, hv0 // HPC, 0)
                        )[(0, None, None, None)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                local = local + 1
        if warp == 0:
            cute.arch.cp_async_bulk_wait_group(0)


@cute.jit
def _launch_sm90(
    state: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    output: cute.Tensor,
    state_indices: cute.Tensor,
    replayssm_rawv: cute.Tensor,
    replayssm_rawk: cute.Tensor,
    replayssm_g: cute.Tensor,
    replayssm_beta: cute.Tensor,
    num_items: Int32,
    grid_ctas: Int32,
    work: cute.Tensor,
    scale: cutlass.Constexpr[float],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    HPC: cutlass.Constexpr[int],
    S: cutlass.Constexpr[int],
    NB: cutlass.Constexpr[int],
    CTAS: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    load = cpasync.CopyBulkTensorTileG2SOp()
    # One checkpoint box is 64 rows x 32 FP32 columns (128 bytes, swizzled).
    tma = cpasync.make_tiled_tma_atom(
        load,
        cute.logical_divide(state, (None, None, _BOX)),
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((1, _D, (_BOX, 1)), stride=(0, _BOX, (1, _D * _BOX))),
        ),
        cta_tiler=(1, _D, _BOX),
    )
    # Q/K: the T rows (128 bytes each, swizzled) of one (batch row, key head).
    qk_box = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3),
        0,
        cute.make_layout((1, T, 1, _D), stride=(0, _D, 0, 1)),
    )
    tma_q = cpasync.make_tiled_tma_atom(load, q, qk_box, cta_tiler=(1, T, 1, _D))
    tma_k = cpasync.make_tiled_tma_atom(load, k, qk_box, cta_tiler=(1, T, 1, _D))
    # V: the T rows of the item's HPC adjacent value heads.
    tma_v = cpasync.make_tiled_tma_atom(
        load,
        v,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((1, T, 1, _D), stride=(0, _D, 0, 1)),
        ),
        cta_tiler=(1, T, 1, _D),
    )
    # Gates: the aligned 8-head (16-byte) chunk of each token.
    ab_box = cute.make_layout((1, T, 8), stride=(0, 8, 1))
    tma_a = cpasync.make_tiled_tma_atom(load, a, ab_box, cta_tiler=(1, T, 8))
    tma_b = cpasync.make_tiled_tma_atom(load, b, ab_box, cta_tiler=(1, T, 8))
    store = cpasync.CopyBulkTensorTileS2GOp()
    tma_rawv = cpasync.make_tiled_tma_atom(
        store,
        replayssm_rawv,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((1, HPC, T, _D), stride=(0, _tpad(T) * _D, _D, 1)),
        ),
        cta_tiler=(1, HPC, T, _D),
    )
    tma_rawk = cpasync.make_tiled_tma_atom(
        store,
        replayssm_rawk,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((1, 1, T, _D), stride=(0, 0, _D, 1)),
        ),
        cta_tiler=(1, 1, T, _D),
    )
    tma_out = cpasync.make_tiled_tma_atom(
        store,
        output,
        cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((1, T, HPC, _D), stride=(0, HPC * _D, _D, 1)),
        ),
        cta_tiler=(1, T, HPC, _D),
    )
    threads = (HPC * _D // 16 + 3) * 32
    gdn_verify_kernel_mtp_replayssm_sm90(
        tma.atom,
        tma.tma_tensor,
        tma_q.atom,
        tma_q.tma_tensor,
        tma_k.atom,
        tma_k.tma_tensor,
        tma_v.atom,
        tma_v.tma_tensor,
        tma_a.atom,
        tma_a.tma_tensor,
        tma_b.atom,
        tma_b.tma_tensor,
        tma_rawv.atom,
        tma_rawv.tma_tensor,
        tma_rawk.atom,
        tma_rawk.tma_tensor,
        tma_out.atom,
        tma_out.tma_tensor,
        A_log,
        dt_bias,
        state_indices,
        replayssm_g,
        replayssm_beta,
        num_items,
        work,
        scale,
        T,
        H,
        HV,
        HPC,
        S,
        NB,
    ).launch(
        grid=(grid_ctas, 1, 1),
        block=(threads, 1, 1),
        max_number_threads=(threads, 1, 1),
        min_blocks_per_mp=CTAS,
        stream=stream,
    )


_CACHE: dict = {}
# Item counters of the dynamic schedule, one pair per (device, stream): the
# kernel claims items from it and its last CTA resets it, so launches in one
# stream (graph replays included) reuse it while concurrent launches in other
# streams get their own.
_WORK: dict = {}


def _slot_dynamic(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tuple(range(tensor.dim())),
        divisibility=1,
    )


def check_sm90_replayssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    output: torch.Tensor,
    T: int,
    H: int,
    HV: int,
) -> None:
    """Raise ``ValueError`` unless the SM90 ReplaySSM verify kernel serves the call.

    ``gated_delta_rule_mtp`` has no other SM90 ReplaySSM path, so calls outside
    this contract are rejected rather than silently rerouted.
    """
    if q.shape[-1] != _D or v.shape[-1] != _D:
        raise ValueError(
            f"SM90 ReplaySSM verify requires K = V = {_D}, "
            f"got K={q.shape[-1]}, V={v.shape[-1]}"
        )
    if T > 16:
        raise ValueError(f"SM90 ReplaySSM verify supports T <= 16, got T={T}")
    if HV % 8 != 0:
        raise ValueError(f"SM90 ReplaySSM verify requires HV % 8 == 0, got HV={HV}")
    if not state.is_contiguous():
        raise ValueError("SM90 ReplaySSM verify requires a contiguous state pool")
    # TMA moves the activations and the output: unit inner stride and 16-byte
    # aligned outer strides (BF16 after the MTP input conversion).
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("a", a),
        ("b", b),
        ("output", output),
    ):
        if tensor.stride(-1) != 1 or any(st % 8 for st in tensor.stride()[:-1]):
            raise ValueError(
                f"SM90 ReplaySSM verify loads {name} with TMA: it needs a unit "
                "inner stride and outer strides that are multiples of 8 elements"
            )


def run_gdn_verify_kernel_mtp_replayssm_sm90(
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    state_indices: torch.Tensor,
    replayssm_rawv: torch.Tensor,
    replayssm_rawk: torch.Tensor,
    replayssm_g: torch.Tensor,
    replayssm_beta: torch.Tensor,
    T: int,
    H: int,
    HV: int,
    scale: float,
) -> None:
    target = gdn_device_target(state.device)
    # A CTA item is the value heads of one Q/K head that share its Q/K and Gram
    # work: two when HV / H is even, one otherwise.
    hpc = 2 if (HV // H) % 2 == 0 else 1
    B = q.shape[0]
    num_items = B * HV // hpc
    ctas, stages, nbuf = _pipeline_config(T, hpc, HV)
    num_sms = torch.cuda.get_device_properties(state.device).multi_processor_count
    grid_ctas = min(num_items, ctas * num_sms)
    layout_key = tuple(
        tuple(i for i, stride in enumerate(t.stride()) if stride == 1)
        for t in (A_log, a, dt_bias, q, k, v, b, output, state_indices)
    )
    key = (
        T,
        H,
        HV,
        hpc,
        stages,
        nbuf,
        ctas,
        scale,
        target.compile_key,
        A_log.dtype,
        dt_bias.dtype,
        state_indices.dtype,
        layout_key,
    )
    cache = _CACHE.setdefault(key, {})
    torch_stream = torch.cuda.current_stream(device=state.device)
    work = _WORK.get((state.device, torch_stream.cuda_stream))
    if work is None:
        work = torch.zeros(2, dtype=torch.int32, device=state.device)
        _WORK[(state.device, torch_stream.cuda_stream)] = work
    stream = cuda.CUstream(torch_stream.cuda_stream)
    if "compiled" not in cache:
        compile_options = gdn_compile_options(
            state.device, cute.EnableTVMFFI(True), cute.GenerateLineInfo(True)
        )
        state_cute = from_dlpack(state, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        cache["compiled"] = build_and_load_cute_dsl_kernel(
            "gdn_replayssm_verify_sm90",
            make_kernel_name("verify_sm90", target.arch, *key[:8], *key[9:]),
            lambda: cute.compile[compile_options](
                _launch_sm90,
                state_cute,
                from_dlpack(A_log, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(a, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(q, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(k, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(v, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(b, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(output, assumed_align=16).mark_layout_dynamic(),
                from_dlpack(state_indices, assumed_align=16).mark_layout_dynamic(),
                _slot_dynamic(replayssm_rawv),
                _slot_dynamic(replayssm_rawk),
                _slot_dynamic(replayssm_g),
                _slot_dynamic(replayssm_beta),
                Int32(num_items),
                Int32(grid_ctas),
                from_dlpack(work),
                scale=scale,
                T=T,
                H=H,
                HV=HV,
                HPC=hpc,
                S=stages,
                NB=nbuf,
                CTAS=ctas,
                stream=stream,
            ),
            extra_key_files=(__file__,),
        )
    cache["compiled"](
        state,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        state_indices,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        num_items,
        grid_ctas,
        work,
        stream,
    )
