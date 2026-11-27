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

"""ReplaySSM commit on SM90 (K = V = 64) through
``gated_delta_rule_replayssm_commit``, alone and after SM90 verify."""

import pytest
import torch

from flashinfer.gdn_decode import (
    gated_delta_rule_mtp,
    gated_delta_rule_replayssm_commit,
)

_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

pytestmark = pytest.mark.skipif(
    _CAPABILITY != (9, 0), reason="SM90 ReplaySSM commit requires SM90"
)

_D = 64
# TF32 contractions against an FP64 recurrence; the same bounds as the SM100
# tcgen05 commit.
_ATOL, _RTOL = 1e-4, 1e-3


def _case(layers, batch, steps, heads, hv, seed=0, extra_slots=3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    slots = batch + extra_slots

    def randn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(dtype)

    perm = torch.randperm(slots, generator=gen, device="cuda").to(torch.int32)
    return dict(
        checkpoint_state=randn(layers, slots, hv, _D, _D, scale=0.1),
        rawv_cache=randn(layers, slots, hv, steps, _D, scale=0.5, dtype=torch.bfloat16),
        rawk_cache=randn(layers, slots, heads, steps, _D, dtype=torch.bfloat16),
        g_cache=-torch.rand(layers, slots, hv, steps, generator=gen, device="cuda"),
        beta_cache=torch.rand(layers, slots, hv, steps, generator=gen, device="cuda"),
        state_indices=perm[:batch].contiguous(),
        accept_lens=torch.randint(
            0, steps + 1, (batch,), generator=gen, device="cuda"
        ).to(torch.int32),
    )


def _reference(case, null_block_id=-1, track_indices=None, track_steps=None):
    """Sequential FP64 replay of each accepted prefix, per request."""
    expected = case["checkpoint_state"].double().clone()
    hv = expected.shape[2]
    heads, steps = case["rawk_cache"].shape[2], case["rawk_cache"].shape[3]
    k = case["rawk_cache"].double()
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    k = k.repeat_interleave(hv // heads, dim=2)
    for n in range(case["state_indices"].shape[0]):
        slot = int(case["state_indices"][n])
        accepted = min(int(case["accept_lens"][n]), steps)
        if slot <= null_block_id or accepted <= 0:
            continue
        state = expected[:, slot].clone()
        for t in range(accepted):
            state = state * case["g_cache"][:, slot, :, t, None, None].double().exp()
            kt = k[:, slot, :, t, None, :]
            prediction = (state * kt).sum(-1)
            delta = (case["rawv_cache"][:, slot, :, t].double() - prediction) * case[
                "beta_cache"
            ][:, slot, :, t, None].double()
            state = state + delta[..., None] * kt
            if (
                track_indices is not None
                and int(track_indices[n]) > null_block_id
                and int(track_steps[n]) == t
            ):
                expected[:, int(track_indices[n])] = state
        expected[:, slot] = state
    return expected


def _commit(case, **kwargs):
    gated_delta_rule_replayssm_commit(
        case["checkpoint_state"],
        case["rawv_cache"],
        case["rawk_cache"],
        case["g_cache"],
        case["beta_cache"],
        case["state_indices"],
        case["accept_lens"],
        **kwargs,
    )


@pytest.mark.parametrize("steps", [1, 3, 4, 7, 8, 9, 16], ids=lambda t: f"T{t}")
@pytest.mark.parametrize("batch", [1, 17, 130], ids=lambda b: f"B{b}")
@pytest.mark.parametrize(
    "layers,heads,hv",
    [(1, 32, 64), (3, 32, 64), (2, 8, 8), (2, 8, 24)],
    ids=["L1_H32_HV64", "L3_H32_HV64", "L2_H8_HV8", "L2_H8_HV24"],
)
def test_sm90_replayssm_commit(layers, batch, steps, heads, hv):
    case = _case(layers, batch, steps, heads, hv, seed=steps * 131 + batch)
    if batch > 1:
        case["state_indices"][1] = -1  # padded request
        case["accept_lens"][0] = steps  # one full accept
    expected = _reference(case)
    _commit(case)
    torch.testing.assert_close(
        case["checkpoint_state"].double(), expected, atol=_ATOL, rtol=_RTOL
    )
    # Slots outside the live requests are bit-identical.
    live = (case["state_indices"] >= 0) & (case["accept_lens"] > 0)
    untouched = torch.ones(expected.shape[1], dtype=torch.bool, device="cuda")
    untouched[case["state_indices"][live].long()] = False
    assert torch.equal(
        case["checkpoint_state"][:, untouched].double(), expected[:, untouched]
    )


@pytest.mark.parametrize("steps", [4, 8, 12, 16], ids=lambda t: f"T{t}")
def test_sm90_replayssm_commit_track(steps):
    batch = 9
    case = _case(2, batch, steps, 32, 64, seed=steps, extra_slots=batch + 2)
    case["accept_lens"] = torch.tensor(
        [steps, 1, steps // 2, 0, steps, 2, steps - 1, 3, steps],
        dtype=torch.int32,
        device="cuda",
    )
    used = set(case["state_indices"].tolist())
    free = [s for s in range(case["checkpoint_state"].shape[1]) if s not in used]
    track_indices = torch.tensor(free[:batch], dtype=torch.int32, device="cuda")
    track_indices[7] = -1  # null destination
    # In range, the first and last accepted step, past the prefix, negative.
    track_steps = torch.tensor(
        [steps // 2, 0, steps // 2 - 1, 0, steps - 1, 2, -1, 1, steps],
        dtype=torch.int32,
        device="cuda",
    )
    expected = _reference(case, -1, track_indices, track_steps)
    _commit(case, track_state_indices=track_indices, track_steps=track_steps)
    torch.testing.assert_close(
        case["checkpoint_state"].double(), expected, atol=_ATOL, rtol=_RTOL
    )


def test_sm90_replayssm_commit_null_block_id():
    case = _case(2, 12, 8, 32, 64, seed=5)
    case["state_indices"][:4] = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    before = case["checkpoint_state"].clone()
    expected = _reference(case, null_block_id=0)
    _commit(case, null_block_id=0)
    torch.testing.assert_close(
        case["checkpoint_state"].double(), expected, atol=_ATOL, rtol=_RTOL
    )
    assert torch.equal(case["checkpoint_state"][:, 0], before[:, 0])


def test_sm90_replayssm_commit_null_block_id_track():
    """Snapshots aimed at slots <= null_block_id are dropped."""
    batch = 6
    case = _case(2, batch, 8, 32, 64, seed=12, extra_slots=batch + 2)
    case["state_indices"] = torch.arange(2, 2 + batch, dtype=torch.int32, device="cuda")
    case["accept_lens"] = torch.full((batch,), 6, dtype=torch.int32, device="cuda")
    track_indices = torch.arange(
        2 + batch, 2 + 2 * batch, dtype=torch.int32, device="cuda"
    )
    track_indices[1] = 0  # the null block
    track_indices[3] = -1
    track_steps = torch.tensor([0, 2, 5, 3, 1, 4], dtype=torch.int32, device="cuda")
    before = case["checkpoint_state"].clone()
    expected = _reference(case, 0, track_indices, track_steps)
    _commit(
        case,
        null_block_id=0,
        track_state_indices=track_indices,
        track_steps=track_steps,
    )
    torch.testing.assert_close(
        case["checkpoint_state"].double(), expected, atol=_ATOL, rtol=_RTOL
    )
    assert torch.equal(case["checkpoint_state"][:, :2], before[:, :2])


def test_sm90_replayssm_commit_noop():
    case = _case(2, 16, 8, 32, 64, seed=6)
    case["accept_lens"].zero_()
    case["state_indices"][::2] = -1
    before = case["checkpoint_state"].clone()
    _commit(case)
    assert torch.equal(case["checkpoint_state"], before)


@pytest.mark.parametrize("steps", [5, 8, 12], ids=lambda t: f"T{t}")
@pytest.mark.parametrize("track", [False, True], ids=["plain", "track"])
def test_sm90_replayssm_commit_rejected_tokens_ignored(steps, track):
    """Window rows past the accepted prefix never reach the checkpoint."""
    batch = 8
    case = _case(1, batch, steps, 32, 64, seed=7 + steps, extra_slots=batch)
    case["accept_lens"] = torch.full((batch,), 3, dtype=torch.int32, device="cuda")
    kwargs = {}
    if track:
        used = set(case["state_indices"].tolist())
        free = [s for s in range(case["checkpoint_state"].shape[1]) if s not in used]
        kwargs = dict(
            track_state_indices=torch.tensor(
                free[:batch], dtype=torch.int32, device="cuda"
            ),
            track_steps=torch.ones(batch, dtype=torch.int32, device="cuda"),
        )
    expected = _reference(
        case, -1, kwargs.get("track_state_indices"), kwargs.get("track_steps")
    )
    case["rawv_cache"][:, :, :, 3:] = float("nan")
    case["rawk_cache"][:, :, :, 3:] = float("nan")
    case["g_cache"][:, :, :, 3:] = float("nan")
    case["beta_cache"][:, :, :, 3:] = float("nan")
    _commit(case, **kwargs)
    torch.testing.assert_close(
        case["checkpoint_state"].double(), expected, atol=_ATOL, rtol=_RTOL
    )


def test_sm90_replayssm_commit_cuda_graph():
    case = _case(2, 16, 8, 32, 64, seed=8)
    eager = {name: t.clone() for name, t in case.items()}
    _commit(eager)
    _commit(_case(2, 16, 8, 32, 64, seed=9))  # compile and warm up
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _commit(case)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(case["checkpoint_state"], eager["checkpoint_state"])


def test_sm90_replayssm_commit_graph_replays():
    """Replays of a launch large enough to use the dynamic item schedule: the
    counter is reset in-kernel, so every replay commits every item once."""
    case = _case(2, 64, 8, 32, 64, seed=13)
    _commit({name: t.clone() for name, t in case.items()})  # compile and warm up
    initial = case["checkpoint_state"].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _commit(case)
    case["checkpoint_state"].copy_(initial)
    expected = case["checkpoint_state"].double()
    for _ in range(3):
        expected = _reference(dict(case, checkpoint_state=expected.float()))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            case["checkpoint_state"].double(), expected, atol=3 * _ATOL, rtol=3 * _RTOL
        )
        expected = case["checkpoint_state"].double()


def test_sm90_replayssm_commit_concurrent_streams():
    """Launches running at the same time in two streams schedule their items
    independently."""
    cases = [_case(2, batch, 8, 32, 64, seed=batch) for batch in (64, 57)]
    expected = [_reference(case) for case in cases]
    streams = [torch.cuda.Stream() for _ in cases]
    torch.cuda.synchronize()
    for case, stream in zip(cases, streams, strict=True):
        with torch.cuda.stream(stream):
            _commit(case)
    torch.cuda.synchronize()
    for case, ref in zip(cases, expected, strict=True):
        torch.testing.assert_close(
            case["checkpoint_state"].double(), ref, atol=_ATOL, rtol=_RTOL
        )


def _verify_inputs(batch, steps, heads, hv, gen):
    def randn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(dtype)

    return dict(
        q=randn(batch, steps, heads, _D),
        k=randn(batch, steps, heads, _D),
        v=randn(batch, steps, hv, _D, scale=0.5),
        a=randn(batch, steps, hv, scale=0.5),
        b=randn(batch, steps, hv),
    )


@pytest.mark.parametrize("steps", [3, 4, 8, 12, 16], ids=lambda t: f"T{t}")
@pytest.mark.parametrize("track", [False, True], ids=["plain", "track"])
def test_sm90_replayssm_verify_commit_cycles(steps, track):
    """Verify (frozen checkpoints) then commit, over several draft windows."""
    batch, heads, hv = 6, 32, 64
    slots = 2 * batch + 1
    gen = torch.Generator(device="cuda").manual_seed(steps + 17 * track)
    params = dict(
        A_log=torch.randn(hv, generator=gen, device="cuda") * 0.3,
        dt_bias=torch.randn(hv, generator=gen, device="cuda") * 0.1,
    )
    checkpoint = (
        torch.randn(1, slots, hv, _D, _D, generator=gen, device="cuda") * 0.1
    ).contiguous()
    windows = dict(
        replayssm_rawv=torch.empty(
            slots, hv, steps, _D, dtype=torch.bfloat16, device="cuda"
        ),
        replayssm_rawk=torch.empty(
            slots, heads, steps, _D, dtype=torch.bfloat16, device="cuda"
        ),
        replayssm_g=torch.empty(slots, hv, steps, device="cuda"),
        replayssm_beta=torch.empty(slots, hv, steps, device="cuda"),
    )
    indices = torch.arange(batch, dtype=torch.int32, device="cuda") * 2
    expected = checkpoint.double().clone()
    for _ in range(3):
        inputs = _verify_inputs(batch, steps, heads, hv, gen)
        before = checkpoint.clone()
        gated_delta_rule_mtp(
            **inputs,
            **params,
            initial_state=checkpoint[0],
            initial_state_indices=indices,
            disable_state_update=True,
            use_qk_l2norm=True,
            scale=_D**-0.5,
            cache_replayssm=True,
            **windows,
        )
        assert torch.equal(checkpoint, before)
        accept = torch.randint(1, steps + 1, (batch,), generator=gen, device="cuda").to(
            torch.int32
        )
        kwargs = {}
        if track:
            kwargs = dict(
                track_state_indices=indices + 1,
                track_steps=torch.randint(
                    0, steps, (batch,), generator=gen, device="cuda"
                ).to(torch.int32),
            )
        case = dict(
            checkpoint_state=checkpoint,
            rawv_cache=windows["replayssm_rawv"][None],
            rawk_cache=windows["replayssm_rawk"][None],
            g_cache=windows["replayssm_g"][None],
            beta_cache=windows["replayssm_beta"][None],
            state_indices=indices,
            accept_lens=accept,
        )
        # Reference from the verify inputs, not from the cached windows.
        k = inputs["k"].double()
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        k = k.repeat_interleave(hv // heads, dim=2)
        log_g = -params["A_log"].double().exp() * torch.nn.functional.softplus(
            inputs["a"].double() + params["dt_bias"].double()
        )
        beta = inputs["b"].double().sigmoid()
        for n in range(batch):
            slot = int(indices[n])
            state = expected[0, slot].clone()
            for t in range(int(accept[n])):
                state = state * log_g[n, t, :, None, None].exp()
                delta = (
                    inputs["v"][n, t].double() - (state * k[n, t, :, None, :]).sum(-1)
                ) * beta[n, t, :, None]
                state = state + delta[..., None] * k[n, t, :, None, :]
                if track and int(kwargs["track_steps"][n]) == t:
                    expected[0, slot + 1] = state
            expected[0, slot] = state
        gated_delta_rule_replayssm_commit(
            case["checkpoint_state"],
            case["rawv_cache"],
            case["rawk_cache"],
            case["g_cache"],
            case["beta_cache"],
            indices,
            accept,
            **kwargs,
        )
        torch.testing.assert_close(checkpoint.double(), expected, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize(
    "invalid",
    [
        "backend_simt",
        "backend_tcgen05",
        "accept_paths",
        "unnormalized",
        "steps_17",
        "head_dim_128",
        "u_cache",
    ],
)
def test_sm90_replayssm_commit_validation(invalid):
    steps = 17 if invalid == "steps_17" else 4
    case = _case(1, 4, steps, 8, 8, seed=11)
    kwargs = {}
    if invalid == "backend_simt":
        kwargs["backend"] = "simt"
    elif invalid == "backend_tcgen05":
        kwargs["backend"] = "tcgen05"
    elif invalid == "accept_paths":
        kwargs["accept_paths"] = torch.zeros(4, steps, dtype=torch.int32, device="cuda")
    elif invalid == "unnormalized":
        kwargs["use_qk_l2norm"] = False
    elif invalid == "head_dim_128":
        slots = case["checkpoint_state"].shape[1]
        case["checkpoint_state"] = torch.zeros(1, slots, 8, 128, 128, device="cuda")
        case["rawv_cache"] = torch.zeros(
            1, slots, 8, steps, 128, dtype=torch.bfloat16, device="cuda"
        )
        case["rawk_cache"] = torch.zeros(
            1, slots, 8, steps, 128, dtype=torch.bfloat16, device="cuda"
        )
    elif invalid == "u_cache":
        kwargs["u_cache"] = torch.zeros(
            1, case["checkpoint_state"].shape[1], 8, steps, _D, device="cuda"
        )
        case["rawv_cache"] = None
        case["beta_cache"] = None
    with pytest.raises(ValueError):
        _commit(case, **kwargs)
