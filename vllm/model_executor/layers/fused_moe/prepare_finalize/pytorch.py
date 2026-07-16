# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.all2all import PyTorchModularAll2AllManager
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    _quantize_and_setup_dispatch,
    _unwrap_scale_and_prepare_for_moe,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)


class PyTorchModularPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    def __init__(
        self,
        manager: PyTorchModularAll2AllManager,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
    ):
        super().__init__()
        self.manager = manager
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        # Returns True because the Manager reduces Top-K dimensions during combine.
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Prepares quantization scales and routes inputs."""

        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )
        extra_tensors = list(scales) if scales is not None else None

        # Execute sparse dispatch routing
        res = self.manager.dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=False,
            extra_tensors=extra_tensors,
        )

        if extra_tensors is None:
            a1q, topk_weights, topk_ids = res
            a1q_scale = a1q_scale_orig
        else:
            a1q, topk_weights, topk_ids, gathered_extras = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(gathered_extras, quant_config)

        # Offset global indices to determine local expert index spaces
        local_topk_ids = topk_ids - self.rank_expert_offset

        # Build local expert token allocations count metadata
        num_local_experts = num_experts // self.dp_size
        local_counts = torch.bincount(
            local_topk_ids.flatten(), minlength=num_local_experts
        )
        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            local_counts.tolist(), device=a1q.device
        )

        return a1q, a1q_scale, expert_tokens_meta, local_topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        """Applies weights before running the reverse combination sequence."""

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()

            # Align local expert offsets back to the global index space
            global_topk_ids = topk_ids + self.rank_expert_offset

            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=global_topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # Re-aggregate, merge representations, and copy results back
        combined_x = self.manager.combine(
            fused_expert_output, is_sequence_parallel=False
        )
        output.copy_(combined_x, non_blocking=True)
