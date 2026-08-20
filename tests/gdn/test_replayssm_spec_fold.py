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

from __future__ import annotations

import pytest
import torch

from flashinfer.gdn_kernels.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ReplaySSM commit requires CUDA"
)


def _reference_fold(
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    num_k_heads: int,
    track_indices: torch.Tensor | None = None,
    track_steps: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    null_block_id: int = -1,
) -> torch.Tensor:
    expected = checkpoint_state.clone()
    num_layers, _, num_value_heads, _, _ = expected.shape
    value_heads_per_key_head = num_value_heads // num_k_heads

    for request in range(state_indices.numel()):
        slot = int(state_indices[request].item())
        num_commit = int(accept_lens[request].item())
        if slot <= null_block_id or num_commit <= 0:
            continue
        for layer in range(num_layers):
            for value_head in range(num_value_heads):
                key_head = value_head // value_heads_per_key_head
                state = expected[layer, slot, value_head].clone()
                for step in range(num_commit):
                    key = rawk_cache[layer, slot, key_head, step].float()
                    if use_qk_l2norm:
                        key = key * torch.rsqrt(torch.sum(key * key) + 1.0e-6)
                    state *= torch.exp(g_cache[layer, slot, value_head, step])
                    prediction = torch.mv(state, key)
                    delta = (
                        rawv_cache[layer, slot, value_head, step].float()
                        - prediction
                    ) * beta_cache[layer, slot, value_head, step]
                    state += delta[:, None] * key[None, :]

                    if track_indices is not None and track_steps is not None:
                        track_slot = int(track_indices[request].item())
                        track_step = int(track_steps[request].item())
                        if track_slot > null_block_id and step == track_step:
                            expected[layer, track_slot, value_head] = state
                expected[layer, slot, value_head] = state
    return expected


@pytest.mark.parametrize("use_qk_l2norm", [True, False])
def test_replayssm_fold_commit_matches_reference(use_qk_l2norm: bool) -> None:
    torch.manual_seed(0)
    device = "cuda"
    num_layers = 3
    num_slots = 8
    num_value_heads = 8
    num_key_heads = 2
    max_cache_len = 8

    checkpoint = (
        torch.randn(
            num_layers,
            num_slots,
            num_value_heads,
            128,
            128,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )
    rawv = (
        torch.randn(
            num_layers,
            num_slots,
            num_value_heads,
            max_cache_len,
            128,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    rawk = (
        torch.randn(
            num_layers,
            num_slots,
            num_key_heads,
            max_cache_len,
            128,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    log_g = -torch.rand(
        num_layers,
        num_slots,
        num_value_heads,
        max_cache_len,
        dtype=torch.float32,
        device=device,
    )
    beta = torch.sigmoid(
        torch.randn(
            num_layers,
            num_slots,
            num_value_heads,
            max_cache_len,
            dtype=torch.float32,
            device=device,
        )
    )

    # Requests 1 and 2 exercise null-state and zero-accept no-ops.  Track
    # destinations are disjoint from every live checkpoint destination.
    state_indices = torch.tensor([0, -1, 2, 3], dtype=torch.int32, device=device)
    accept_lens = torch.tensor([6, 4, 0, 2], dtype=torch.int32, device=device)
    track_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=device)
    track_steps = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device=device)

    expected = _reference_fold(
        checkpoint,
        rawv,
        rawk,
        log_g,
        beta,
        state_indices,
        accept_lens,
        num_key_heads,
        track_indices,
        track_steps,
        use_qk_l2norm,
    )
    actual = checkpoint.clone()
    commit_gdn_replayssm_fold_all_layers(
        checkpoint_state=actual,
        rawv_cache=rawv,
        rawk_cache=rawk,
        g_cache=log_g,
        beta_cache=beta,
        ssm_state_indices=state_indices,
        accept_lens=accept_lens,
        max_cache_len=max_cache_len,
        num_k_heads=num_key_heads,
        mamba_track_indices=track_indices,
        mamba_steps_to_track=track_steps,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
