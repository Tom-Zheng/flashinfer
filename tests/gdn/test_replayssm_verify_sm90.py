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

"""ReplaySSM verify on SM90 (K = V = 64) through ``gated_delta_rule_mtp``."""

import pytest
import torch

from flashinfer.gdn_decode import gated_delta_rule_mtp

_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

pytestmark = pytest.mark.skipif(
    _CAPABILITY != (9, 0), reason="SM90 ReplaySSM verify requires SM90"
)

_D = 64


def _inputs(batch, steps, heads, value_heads, dtype=torch.bfloat16):
    torch.manual_seed(1234 + 97 * batch + steps)
    device = "cuda"
    slots = batch + 3

    def rand(*shape, dtype=dtype, scale=1.0):
        return torch.randn(*shape, dtype=dtype, device=device) * scale

    # Slots in reverse order, the last request padded (-1).
    indices = torch.arange(batch, 0, -1, dtype=torch.int32, device=device)
    if batch > 1:
        indices[-1] = -1
    args = dict(
        q=rand(batch, steps, heads, _D),
        k=rand(batch, steps, heads, _D),
        v=rand(batch, steps, value_heads, _D, scale=0.5),
        initial_state=rand(slots, value_heads, _D, _D, dtype=torch.float32, scale=0.1),
        initial_state_indices=indices,
        A_log=rand(value_heads, dtype=torch.float32, scale=0.3),
        a=rand(batch, steps, value_heads, scale=0.5),
        dt_bias=rand(value_heads, dtype=torch.float32, scale=0.1),
        b=rand(batch, steps, value_heads),
        disable_state_update=True,
        use_qk_l2norm=True,
        scale=_D**-0.5,
    )

    def window(*shape, dtype):
        return torch.full(shape, 7.0, dtype=dtype, device=device)

    cache = dict(
        replayssm_rawv=window(slots, value_heads, steps, _D, dtype=torch.bfloat16),
        replayssm_rawk=window(slots, heads, steps, _D, dtype=torch.bfloat16),
        replayssm_g=window(slots, value_heads, steps, dtype=torch.float32),
        replayssm_beta=window(slots, value_heads, steps, dtype=torch.float32),
    )
    return args, cache


def _reference(args):
    """Sequential FP64 delta-rule recurrence, independent of the kernel's math."""
    q, k, v, a, b = (
        args[n].to(torch.bfloat16).double() for n in ("q", "k", "v", "a", "b")
    )
    heads, hv = q.shape[2], v.shape[2]
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * args["scale"]
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(hv // heads, dim=2)
    k = k.repeat_interleave(hv // heads, dim=2)
    log_g = -args["A_log"].double().exp() * torch.nn.functional.softplus(
        a + args["dt_bias"].double()
    )
    beta = b.sigmoid()
    slots = args["initial_state_indices"].long().clamp_min(0)
    state = args["initial_state"][slots].double()
    outputs = []
    for t in range(q.shape[1]):
        state = state * log_g[:, t, :, None, None].exp()
        prediction = (state * k[:, t, :, None, :]).sum(-1)
        delta = (v[:, t] - prediction) * beta[:, t, :, None]
        state = state + delta[..., None] * k[:, t, :, None, :]
        outputs.append((state * q[:, t, :, None, :]).sum(-1))
    return torch.stack(outputs, 1), log_g, beta


def _check(args, cache, out):
    before = args["initial_state"].clone()
    expected, log_g, beta = _reference(args)
    output, state = gated_delta_rule_mtp(
        **args, output=out, cache_replayssm=True, **cache
    )
    assert output is out and state is args["initial_state"]
    live = args["initial_state_indices"] >= 0
    # TF32 tensor-core contractions against an FP64 recurrence.
    torch.testing.assert_close(out[live].double(), expected[live], atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(
        out[~live], torch.full_like(out[~live], 3.0), atol=0, rtol=0
    )
    torch.testing.assert_close(state, before, atol=0, rtol=0)
    slots = args["initial_state_indices"][live].long()
    for name, ref in (
        ("replayssm_rawv", args["v"].to(torch.bfloat16).permute(0, 2, 1, 3)),
        ("replayssm_rawk", args["k"].to(torch.bfloat16).permute(0, 2, 1, 3)),
        ("replayssm_g", log_g.permute(0, 2, 1)),
        ("replayssm_beta", beta.permute(0, 2, 1)),
    ):
        tol = 0 if "raw" in name else 2e-6
        torch.testing.assert_close(
            cache[name][slots].double(), ref[live].double(), atol=tol, rtol=tol
        )
        untouched = torch.ones(cache[name].shape[0], device="cuda", dtype=torch.bool)
        untouched[slots] = False
        torch.testing.assert_close(
            cache[name][untouched],
            torch.full_like(cache[name][untouched], 7.0),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("steps", [3, 4, 5, 7, 8, 9, 12, 16], ids=lambda t: f"T{t}")
@pytest.mark.parametrize("batch", [1, 17, 300], ids=lambda b: f"B{b}")
@pytest.mark.parametrize(
    "heads,hv",
    [(32, 64), (8, 8), (8, 24), (2, 8)],
    ids=["H32_HV64", "H8_HV8", "H8_HV24", "H2_HV8"],
)
def test_sm90_replayssm_verify(batch, steps, heads, hv):
    args, cache = _inputs(batch, steps, heads, hv)
    _check(args, cache, torch.full_like(args["v"], 3.0))


@pytest.mark.parametrize(
    "variant",
    [
        "fp16",
        "fp32_output",
        "int64_indices",
        "bf16_bias",
        "strided",
        "extreme_gates",
        "zero_qk",
        "all_padding",
        "middle_padding",
        "scale",
    ],
)
@pytest.mark.parametrize("steps", [4, 16], ids=lambda t: f"T{t}")
def test_sm90_replayssm_verify_variants(steps, variant):
    args, cache = _inputs(
        5, steps, 32, 64, torch.float16 if variant == "fp16" else torch.bfloat16
    )
    out = torch.full_like(args["v"], 3.0)
    if variant == "fp32_output":
        out = out.float()
    if variant == "int64_indices":
        args["initial_state_indices"] = args["initial_state_indices"].long()
    if variant == "bf16_bias":
        args["dt_bias"] = args["dt_bias"].bfloat16()
    if variant == "strided":
        # Row-padded views: unit inner stride, 16-byte aligned outer strides.
        for name in ("q", "k", "v", "a", "b"):
            value = args[name]
            storage = torch.empty(
                (*value.shape[:-1], value.shape[-1] + 8),
                dtype=value.dtype,
                device="cuda",
            )
            args[name] = storage[..., : value.shape[-1]]
            args[name].copy_(value)
    if variant == "extreme_gates":
        args["a"][:, 0].fill_(100.0)
        args["a"][:, 1].fill_(-30.0)
        args["b"][:, 0].fill_(-100.0)
        args["b"][:, 1].fill_(100.0)
    if variant == "zero_qk":
        args["q"].zero_()
        args["k"].zero_()
    if variant == "all_padding":
        args["initial_state_indices"].fill_(-1)
    if variant == "middle_padding":
        args["initial_state_indices"][2] = -1
    if variant == "scale":
        # The scale is a compile-time constant: a second value must not reuse
        # the default-scale kernel.
        args["scale"] = 0.3
    _check(args, cache, out)


def test_sm90_replayssm_verify_cuda_graph():
    args, cache = _inputs(9, 8, 32, 64)
    out = torch.full_like(args["v"], 3.0)
    gated_delta_rule_mtp(**args, output=out, cache_replayssm=True, **cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gated_delta_rule_mtp(**args, output=out, cache_replayssm=True, **cache)
    fresh, _ = _inputs(9, 8, 32, 64)
    for name in ("q", "k", "v", "a", "b", "initial_state"):
        args[name].copy_(fresh[name])
    for window in cache.values():
        window.fill_(7.0)
    out.fill_(3.0)
    graph.replay()
    torch.cuda.synchronize()
    expected, _, _ = _reference(args)
    live = args["initial_state_indices"] >= 0
    torch.testing.assert_close(out[live].double(), expected[live], atol=1e-3, rtol=1e-2)


def test_sm90_replayssm_verify_graph_replays():
    """Replays of a launch large enough to use the dynamic item schedule:
    its counter is reset in-kernel, so every replay covers every item once."""
    args, cache = _inputs(300, 8, 32, 64)
    out = torch.full_like(args["v"], 3.0)
    gated_delta_rule_mtp(**args, output=out, cache_replayssm=True, **cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gated_delta_rule_mtp(**args, output=out, cache_replayssm=True, **cache)
    gen = torch.Generator(device="cuda").manual_seed(5)
    live = args["initial_state_indices"] >= 0
    for _ in range(3):
        for name in ("q", "k", "v", "a", "b"):
            args[name].copy_(
                torch.randn(args[name].shape, generator=gen, device="cuda")
            )
        out.fill_(3.0)
        graph.replay()
        torch.cuda.synchronize()
        expected, _, _ = _reference(args)
        torch.testing.assert_close(
            out[live].double(), expected[live], atol=1e-3, rtol=1e-2
        )


def test_sm90_replayssm_verify_concurrent_streams():
    """Launches running at the same time in two streams schedule their items
    independently."""
    problems = [_inputs(batch, 8, 32, 64) for batch in (300, 257)]
    outs = [torch.full_like(args["v"], 3.0) for args, _ in problems]
    streams = [torch.cuda.Stream() for _ in problems]
    torch.cuda.synchronize()
    for _ in range(3):
        for (args, cache), out, stream in zip(problems, outs, streams, strict=True):
            with torch.cuda.stream(stream):
                gated_delta_rule_mtp(**args, output=out, cache_replayssm=True, **cache)
    torch.cuda.synchronize()
    for (args, _), out in zip(problems, outs, strict=True):
        expected, _, _ = _reference(args)
        live = args["initial_state_indices"] >= 0
        torch.testing.assert_close(
            out[live].double(), expected[live], atol=1e-3, rtol=1e-2
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "head_dim_128",
        "steps_17",
        "unnormalized",
        "inner_strided",
        "hv_not_multiple_of_8",
        "strided_state",
        "u_mode",
    ],
)
def test_sm90_replayssm_verify_validation(invalid):
    if invalid == "head_dim_128":
        args, cache = _inputs(2, 4, 2, 8)
        for name in ("q", "k"):
            args[name] = torch.randn(2, 4, 2, 128, dtype=torch.bfloat16, device="cuda")
        args["v"] = torch.randn(2, 4, 8, 128, dtype=torch.bfloat16, device="cuda")
        args["initial_state"] = torch.zeros(5, 8, 128, 128, device="cuda")
        cache["replayssm_rawv"] = torch.zeros(
            5, 8, 4, 128, dtype=torch.bfloat16, device="cuda"
        )
        cache["replayssm_rawk"] = torch.zeros(
            5, 2, 4, 128, dtype=torch.bfloat16, device="cuda"
        )
        match = "K = V = 64"
    elif invalid == "steps_17":
        args, cache = _inputs(2, 17, 2, 8)
        match = "T <= 16"
    elif invalid == "unnormalized":
        args, cache = _inputs(2, 4, 2, 8)
        args["use_qk_l2norm"] = False
        match = "use_qk_l2norm"
    elif invalid == "inner_strided":
        args, cache = _inputs(2, 4, 2, 8)
        storage = torch.empty(2, 4, 2, 2 * _D, dtype=torch.bfloat16, device="cuda")
        storage[..., ::2] = args["q"]
        args["q"] = storage[..., ::2]
        match = "TMA"
    elif invalid == "hv_not_multiple_of_8":
        args, cache = _inputs(2, 4, 2, 4)
        match = "HV % 8"
    elif invalid == "strided_state":
        args, cache = _inputs(2, 4, 2, 8)
        state = args["initial_state"]
        storage = torch.zeros((state.shape[0] * 2, *state.shape[1:]), device="cuda")
        args["initial_state"] = storage[::2]
        match = "contiguous state pool"
    else:
        args, cache = _inputs(2, 4, 2, 8)
        cache["replayssm_u"] = torch.zeros(5, 8, 4, _D, device="cuda")
        del cache["replayssm_rawv"], cache["replayssm_beta"]
        match = "SM100"
    with pytest.raises(ValueError, match=match):
        gated_delta_rule_mtp(**args, cache_replayssm=True, **cache)
