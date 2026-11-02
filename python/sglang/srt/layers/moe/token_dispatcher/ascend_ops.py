"""Ascend Ops MoE communication dispatcher.

Expert regrouping is handled by the Ascend runner hooks. This module owns the
AllGather, AllToAllV, MC2, and MegaMoE dispatch boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.moe.routing import (
    INT8_QUANT_MODE,
    MXFP8_QUANT_MODE,
    ROUTING_QUANT_MODES,
    normalize_mxfp_scale,
    npu_moe_finalize_routing,
    npu_moe_init_routing,
)
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import AscendOpsConfig, DispatcherOutputDtype
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu

if is_npu():
    import torch_npu

# Routing and MC2 use different quant-mode enums.
_MC2_QUANT_MODE = {
    DispatcherOutputDtype.BF16: 0,
    DispatcherOutputDtype.INT8: 2,
    DispatcherOutputDtype.MXFP8: 4,
}

_QUANT_MODES = {
    "routing": ROUTING_QUANT_MODES,
    "mc2": _MC2_QUANT_MODE,
}

# MC2 expects an ATen ScalarType integer for ``y_dtype``.
_TORCH_DTYPE_TO_ATEN = {
    torch.int8: 1,
    torch.int32: 3,
    torch.float16: 5,
    torch.float32: 6,
    torch.bfloat16: 15,
    torch.bits8: 21,
    torch.float8_e5m2: 23,
    torch.float8_e4m3fn: 24,
}


class AscendOpsDispatchOutput(NamedTuple):
    """Dispatcher output shared by the Ascend execution paths."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    expert_tokens: Optional[torch.Tensor]
    group_list_type: int
    exec_mode: str
    forward_state: Optional[AscendOpsForwardState]
    quant_mode: int = -1
    recv_counts: Optional[torch.Tensor] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.ASCEND_OPS


class AscendOpsCombineInput(NamedTuple):
    """Expert output consumed by the matching Ascend Ops combine path."""

    hidden_states: torch.Tensor
    forward_state: Optional[AscendOpsForwardState] = None

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.ASCEND_OPS


@dataclass
class _MC2State:
    """Per-forward metadata required by MC2 combine."""

    num_tokens: int
    assist_info_for_combine: torch.Tensor
    ep_send_counts: torch.Tensor
    x_active_mask: Optional[torch.Tensor]
    expand_scales: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass
class _AllGatherState:
    group: Any
    row_sizes: tuple[int, ...]
    input_dtype: torch.dtype
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None
    expanded_row_idx: Optional[torch.Tensor] = None
    expert_tokens: Optional[torch.Tensor] = None
    num_routes: Optional[int] = None


@dataclass
class _AllToAllVState:
    group: Any
    input_splits: list[int]
    output_splits: list[int]
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expanded_row_idx: torch.Tensor
    hidden_shape: torch.Size
    padded_hidden_shape: torch.Size
    local_expanded_row_idx: Optional[torch.Tensor] = None
    local_num_valid_rows: Optional[int] = None


@dataclass
class _MegaMoEState:
    """Per-forward metadata required by the fused MegaMoE runner path."""

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    group: Any
    quant_config: Optional[dict] = None
    num_valid_rows: Optional[int] = None
    dtype_mode: Optional[str] = None


@dataclass(frozen=True)
class _RowLayout:
    """Rank-major token layout shared by Ascend prefill paths."""

    row_sizes: tuple[int, ...]
    local_rows: int
    target_rows: int


@dataclass(frozen=True)
class _PreparedPrefillBatch:
    """Common row layout and inputs prepared for an Ascend prefill path."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    layout: _RowLayout
    num_valid_rows: int


AscendOpsForwardState = _MC2State | _AllGatherState | _AllToAllVState | _MegaMoEState


class AscendOpsDispatcher(BaseDispatcher):
    """Select and execute an Ascend dispatch/combine path."""

    def __init__(
        self,
        moe_runner_config: MoeRunnerConfig,
        moe_ascend_ops_config: Optional[AscendOpsConfig] = None,
    ) -> None:
        super().__init__()
        self.moe_ascend_ops_config = moe_ascend_ops_config or AscendOpsConfig()
        parallel = get_parallel()
        self.tp_group = parallel.tp_group.device_group
        self.ep_coordinator = parallel.moe_ep_group
        self.ep_group = self.ep_coordinator.device_group
        self.ep_size = self.ep_coordinator.world_size
        self.ep_rank = self.ep_coordinator.rank_in_group
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.top_k = moe_runner_config.top_k
        self.hidden_size = moe_runner_config.hidden_size
        self.quant_config: Optional[dict] = None
        self._ep_comm_name: Optional[str] = None
        self._pad_buffers: dict[tuple, tuple] = {}
        if moe_runner_config.num_fused_shared_experts:
            raise ValueError(
                "Ascend Ops does not support fused shared experts; "
                "disable shared-expert fusion before selecting ascend_ops."
            )
        if self.moe_ascend_ops_config.prefill == "allgather":
            if not (
                parallel.tp_size == parallel.moe_ep_size
                and self.num_local_experts * self.ep_size == self.num_experts
                and not moe_runner_config.num_fused_shared_experts
            ):
                raise ValueError(
                    "NPU AllGather requires TP=EP, contiguous expert placement "
                    "and no fused shared experts"
                )

    def set_quant_config(self, quant_config: dict) -> None:
        """Store quantization settings used by the routing kernels."""

        self.quant_config = quant_config

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        topk_weights, topk_ids, _ = topk_output
        if get_is_extend_in_batch():
            if self.moe_ascend_ops_config.prefill == "fused_op":
                return self._dispatch_fused_op(
                    hidden_states, topk_weights, topk_ids
                )
            if self.moe_ascend_ops_config.prefill == "allgather":
                return self._dispatch_all_gather(
                    hidden_states, topk_weights, topk_ids, self.tp_group
                )
            return self._dispatch_all_to_all_v(
                hidden_states, topk_weights, topk_ids, self.tp_group
            )

        if self.moe_ascend_ops_config.decode == "fused_op":
            return self._dispatch_fused_op(
                hidden_states, topk_weights, topk_ids
            )
        return self._dispatch_low_latency(hidden_states, topk_weights, topk_ids)

    def _dispatch_fused_op(self, hidden_states, topk_weights, topk_ids):
        is_extend = get_is_extend_in_batch()
        dp_enabled = is_dp_attention_enabled()
        group = (
            self.ep_group
            if not is_extend or self.moe_ascend_ops_config.decode == "fused_op"
            else self.tp_group
        )
        prepared = self._prepare_prefill_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            getattr(self, "tp_group", None),
            pad_to_target=dp_enabled,
            allow_cp=not is_extend,
            resolve_layout=dp_enabled or is_extend,
        )
        hidden_states = prepared.hidden_states
        topk_weights = prepared.topk_weights
        topk_ids = prepared.topk_ids

        topk_weights = topk_weights.to(torch.float32).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()
        return AscendOpsDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_tokens=None,
            group_list_type=1,
            exec_mode="fused_op",
            forward_state=_MegaMoEState(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                group=group,
                quant_config=(
                    dict(self.quant_config) if self.quant_config is not None else None
                ),
                num_valid_rows=prepared.num_valid_rows,
                dtype_mode=("normal" if is_extend else "low_latency"),
            ),
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if combine_input.format != CombineInputFormat.ASCEND_OPS:
            raise TypeError(
                "AscendOpsDispatcher.combine expects an AscendOpsCombineInput, "
                f"got {combine_input.format.value}"
            )
        if combine_input.forward_state is None:
            return combine_input.hidden_states
        state = combine_input.forward_state
        if isinstance(state, _AllGatherState):
            return self._combine_all_gather(combine_input.hidden_states, state)
        if isinstance(state, _AllToAllVState):
            return self._combine_all_to_all_v(combine_input.hidden_states, state)
        if isinstance(state, _MC2State):
            return self._combine_mc2(combine_input.hidden_states, state)
        if isinstance(state, _MegaMoEState):
            return self._combine_megamoe(combine_input.hidden_states, state)
        raise TypeError(f"unknown Ascend Ops forward state: {type(state)!r}")

    # Decode (MC2 low latency)

    def _dispatch_low_latency(self, hidden_states, topk_weights, topk_ids):
        num_tokens = hidden_states.shape[0]
        pad_bs = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        if pad_bs and num_tokens > pad_bs:
            raise ValueError(
                f"ascend_ops decode got {num_tokens} tokens on this rank but "
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK limits the "
                f"padded batch to {pad_bs}; "
                "raise it or lower the decode batch size."
            )
        return self._dispatch_mc2(
            hidden_states,
            topk_ids,
            topk_weights,
            padded_bs=pad_bs or num_tokens,
            dtype_mode="low_latency",
        )

    def _combine_megamoe(
        self, hidden_states: torch.Tensor, state: _MegaMoEState
    ) -> torch.Tensor:
        """Return the output already combined by the MegaMoE operator."""

        return hidden_states

    def _dispatch_mc2(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        padded_bs: int,
        dtype_mode: str,
    ) -> AscendOpsDispatchOutput:
        """Dispatch one decode batch through MC2."""
        num_tokens = hidden_states.shape[0]
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.float()
        quant_mode = self._resolve_quant_mode("mc2", dtype_mode)

        x_active_mask = None
        if padded_bs != num_tokens:
            hidden_states, topk_ids, topk_weights, x_active_mask = self._pad_dispatch_batch(
                hidden_states, topk_ids, topk_weights, num_tokens, padded_bs
            )

        mxfp8_args = (
            {"y_dtype": _TORCH_DTYPE_TO_ATEN[torch.float8_e4m3fn]}
            if quant_mode == _MC2_QUANT_MODE[DispatcherOutputDtype.MXFP8]
            else {}
        )

        (
            expand_x,
            dynamic_scales,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            _tp_recv_counts,
            _expand_scales,
        ) = torch_npu.npu_moe_distribute_dispatch_v2(
            hidden_states,
            topk_ids,
            self._get_ep_comm_name(),
            self.ep_size,
            self.ep_rank,
            self.num_experts,
            x_active_mask=x_active_mask,
            quant_mode=quant_mode,
            global_bs=0,
            expert_token_nums_type=1,
            comm_alg=self.moe_ascend_ops_config.ll_dispatch_alg,
            **mxfp8_args,
        )

        if quant_mode == _MC2_QUANT_MODE[DispatcherOutputDtype.MXFP8]:
            if hidden_states.shape[1] % 64 != 0:
                raise ValueError(
                    "MC2 MXFP8 decode requires hidden_size to be a multiple "
                    f"of 64, got {hidden_states.shape[1]}"
                )
            # Convert the flat e8m0 scale to the runner's pair-split layout.
            dynamic_scales = dynamic_scales.reshape(
                dynamic_scales.shape[0], hidden_states.shape[1] // 64, 2
            )

        expand_scales = None

        state = _MC2State(
            num_tokens=num_tokens,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_recv_counts,
            x_active_mask=x_active_mask,
            expand_scales=expand_scales,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        return AscendOpsDispatchOutput(
            hidden_states=expand_x,
            hidden_states_scale=(
                dynamic_scales
                if quant_mode != _MC2_QUANT_MODE[DispatcherOutputDtype.BF16]
                else None
            ),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_tokens=expert_token_nums,
            group_list_type=1,
            exec_mode="mc2",
            forward_state=state,
        )

    def _pad_dispatch_batch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens: int,
        padded_bs: int,
    ):
        """Copy a possibly shorter batch into reusable padded buffers."""
        key = (
            padded_bs,
            hidden_states.shape[1],
            topk_ids.shape[1],
            hidden_states.dtype,
            topk_ids.dtype,
            topk_weights.dtype,
        )
        bufs = self._pad_buffers.get(key)
        if bufs is None:
            bufs = (
                hidden_states.new_zeros((padded_bs, hidden_states.shape[1])),
                topk_ids.new_zeros((padded_bs, topk_ids.shape[1])),
                topk_weights.new_zeros((padded_bs, topk_weights.shape[1])),
                torch.zeros(padded_bs, dtype=torch.bool, device=hidden_states.device),
                0,
            )
        padded_x, padded_ids, padded_weights, mask, filled = bufs

        padded_x[:num_tokens] = hidden_states
        padded_ids[:num_tokens] = topk_ids
        padded_weights[:num_tokens] = topk_weights

        mask[:num_tokens] = True
        if filled > num_tokens:
            padded_x[num_tokens:filled] = 0
            padded_ids[num_tokens:filled] = 0
            padded_weights[num_tokens:filled] = 0
            mask[num_tokens:filled] = False

        self._pad_buffers[key] = (
            padded_x,
            padded_ids,
            padded_weights,
            mask,
            num_tokens,
        )
        return padded_x, padded_ids, padded_weights, mask

    def _combine_mc2(
        self, hidden_states: torch.Tensor, state: _MC2State
    ) -> torch.Tensor:
        """Run ``npu_moe_distribute_combine_v2`` for the batch dispatched last."""
        combined = torch_npu.npu_moe_distribute_combine_v2(
            hidden_states,
            state.topk_ids,
            state.assist_info_for_combine,
            state.ep_send_counts,
            state.topk_weights,
            self._get_ep_comm_name(),
            self.ep_size,
            self.ep_rank,
            self.num_experts,
            x_active_mask=state.x_active_mask,
            expand_scales=state.expand_scales,
            global_bs=0,
            comm_quant_mode=0,
            comm_alg=self.moe_ascend_ops_config.ll_combine_alg,
        )

        return combined[: state.num_tokens]

    def _resolve_quant_mode(self, op: str, dtype_mode: Optional[str] = None) -> int:
        """Quant mode op ``"routing"`` or ``"mc2"`` uses for the wire dtype."""
        table = _QUANT_MODES.get(op)
        if table is None:
            raise ValueError(
                f"unknown Ascend Ops quant-mode op {op!r}; "
                f"expected one of {tuple(_QUANT_MODES)}"
            )
        dtype = self._resolve_dispatch_dtype(dtype_mode)
        try:
            return table[dtype]
        except KeyError:
            raise NotImplementedError(
                f"ascend_ops {op} dispatch does not support dispatcher output "
                f"dtype {dtype.value}; supported: "
                f"{tuple(d.value for d in table)}"
            ) from None

    def _resolve_dispatch_dtype(
        self, dtype_mode: Optional[str] = None
    ) -> DispatcherOutputDtype:
        """Resolve a mode-specific dtype, falling back to the shared dtype."""
        quant_config = self.quant_config or {}
        raw = None
        if dtype_mode is not None:
            raw = quant_config.get(f"{dtype_mode}_dispatcher_output_dtype")
        if raw is None:
            raw = quant_config.get("dispatcher_output_dtype")
        if raw is None:
            return DispatcherOutputDtype.BF16
        return DispatcherOutputDtype(raw)

    def _get_ep_comm_name(self) -> str:
        if self._ep_comm_name is None:
            backend = self.ep_group._get_backend(torch.device("npu"))
            if hasattr(backend, "get_hccl_comm_name"):
                name = backend.get_hccl_comm_name(self.ep_rank)
            else:
                name = self.ep_group.get_hccl_comm_name(self.ep_rank)
            self._ep_comm_name = name
        return self._ep_comm_name

    # Prefill: AllGather

    @staticmethod
    def _all_gather_row_sizes(local_rows: int, group, device) -> list[int]:
        """Exchange the actual token rows contributed by every TP rank."""

        world_size = dist.get_world_size(group)
        local_rows_tensor = torch.tensor([local_rows], dtype=torch.int64, device=device)
        gathered_rows = [torch.empty_like(local_rows_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_rows, local_rows_tensor, group=group)
        return [int(rows.item()) for rows in gathered_rows]

    @staticmethod
    def _pad_rows(
        tensor: torch.Tensor, target_rows: int, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pad the leading row dimension for an equal-size collective."""
        if tensor.shape[0] == target_rows:
            return tensor.contiguous()
        if tensor.shape[0] > target_rows:
            raise ValueError(
                f"cannot pad {tensor.shape[0]} rows down to {target_rows}"
            )
        padded = (
            tensor.new_zeros((target_rows, *tensor.shape[1:]))
            if out is None
            else out
        )
        padded.zero_()
        padded[: tensor.shape[0]].copy_(tensor)
        return padded

    @staticmethod
    def _restore_rows(
        gathered: torch.Tensor, row_sizes: tuple[int, ...], max_rows: int
    ) -> torch.Tensor:
        """Remove per-rank padding while preserving rank-major row order."""
        if min(row_sizes) == max_rows:
            return gathered
        return torch.cat(
            [
                gathered[rank * max_rows : rank * max_rows + rows]
                for rank, rows in enumerate(row_sizes)
            ],
            dim=0,
        )

    def _resolve_row_layout(
        self,
        local_rows: int,
        group,
        device: torch.device,
        *,
        allow_cp: bool,
    ) -> _RowLayout:
        """Resolve the rank-major rows once for all prefill execution paths.

        CP=1 uses scheduler metadata and performs no row-count collective. CP
        paths may have zigzag/interleaved row counts, so they exchange the
        actual local count when the caller allows that fallback.
        """
        parallel = get_parallel()
        group_size = dist.get_world_size(group)
        if (
            getattr(parallel, "attn_dp_size", 1) == 1
            and parallel.attn_cp_size == 1
        ):
            row_sizes = (local_rows,) * group_size
            return _RowLayout(
                row_sizes=row_sizes,
                local_rows=local_rows,
                target_rows=local_rows,
            )

        global_rows = get_dp_global_num_tokens()
        if parallel.attn_cp_size == 1 and global_rows is not None:
            attn_tp_size = parallel.attn_tp_size
            if len(global_rows) * attn_tp_size != group_size:
                raise ValueError(
                    "Ascend Ops DP token metadata does not match the TP group: "
                    f"metadata has {len(global_rows)} DP rows, "
                    f"attn_tp_size={attn_tp_size}, group size={group_size}"
                )
            row_sizes = []
            for rows in global_rows:
                base, extra = divmod(rows, attn_tp_size)
                row_sizes.extend(
                    base + int(rank < extra) for rank in range(attn_tp_size)
                )
            rank = dist.get_rank(group)
            if row_sizes[rank] != local_rows:
                raise ValueError(
                    "Ascend Ops local rows disagree with DP token metadata: "
                    f"rank has {local_rows}, expected {row_sizes[rank]}"
                )
            return _RowLayout(
                row_sizes=tuple(row_sizes),
                local_rows=row_sizes[rank],
                target_rows=max(row_sizes),
            )

        if not allow_cp:
            raise RuntimeError(
                "Ascend Ops MegaMoE with attention DP requires CP=1 DP token metadata"
            )
        row_sizes = self._all_gather_row_sizes(local_rows, group, device)
        rank = dist.get_rank(group)
        return _RowLayout(
            row_sizes=tuple(row_sizes),
            local_rows=row_sizes[rank],
            target_rows=max(row_sizes),
        )

    def _prepare_prefill_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        group,
        *,
        pad_to_target: bool,
        allow_cp: bool,
        resolve_layout: bool = True,
    ) -> _PreparedPrefillBatch:
        """Resolve row layout and optionally pad inputs for fused MegaMoE.

        AllGather keeps local rows intact and pads inside its collective,
        whereas MegaMoE requires every rank to enter with the same capacity.
        Both paths therefore share the same DP/TP layout preparation while
        retaining their distinct communication behavior.
        """
        if resolve_layout:
            layout = self._resolve_row_layout(
                hidden_states.shape[0],
                group,
                hidden_states.device,
                allow_cp=allow_cp,
            )
        else:
            local_rows = hidden_states.shape[0]
            layout = _RowLayout((local_rows,), local_rows, local_rows)
        num_valid_rows = layout.local_rows
        if not pad_to_target or hidden_states.shape[0] == layout.target_rows:
            return _PreparedPrefillBatch(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layout=layout,
                num_valid_rows=num_valid_rows,
            )

        if hidden_states.shape[0] > layout.target_rows:
            raise ValueError(
                "Ascend Ops local rows exceed the resolved DP/TP capacity: "
                f"{hidden_states.shape[0]} > {layout.target_rows}"
            )

        (
            hidden_states,
            topk_ids,
            topk_weights,
            _,
        ) = self._pad_dispatch_batch(
            hidden_states,
            topk_ids,
            topk_weights,
            hidden_states.shape[0],
            layout.target_rows,
        )
        if num_valid_rows < hidden_states.shape[0]:
            topk_weights = topk_weights.clone()
            topk_weights[num_valid_rows:].zero_()
        return _PreparedPrefillBatch(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            layout=layout,
            num_valid_rows=num_valid_rows,
        )

    def _dispatch_all_gather(self, hidden_states, topk_weights, topk_ids, group):
        """Gather the full batch on every TP rank."""
        prepared = self._prepare_prefill_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            group,
            pad_to_target=False,
            allow_cp=True,
        )
        hidden_states = prepared.hidden_states
        topk_weights = prepared.topk_weights
        topk_ids = prepared.topk_ids
        input_dtype = hidden_states.dtype
        quant_mode = self._resolve_quant_mode("routing")
        if quant_mode not in (-1, INT8_QUANT_MODE, MXFP8_QUANT_MODE):
            raise NotImplementedError(
                "NPU AllGather supports BF16, INT8 and MXFP8 routing"
            )
        layout = prepared.layout
        row_sizes = layout.row_sizes
        world_size = len(row_sizes)
        max_rows = layout.target_rows
        if max_rows == 0:
            raise RuntimeError("NPU prefill requires at least one global token")

        def gather_rows(tensor):
            padded = self._pad_rows(tensor, max_rows)
            gathered = tensor.new_empty((world_size * max_rows, *tensor.shape[1:]))
            dist.all_gather_into_tensor(gathered, padded, group=group)
            return self._restore_rows(gathered, row_sizes, max_rows)

        scales = None
        if quant_mode == MXFP8_QUANT_MODE:
            if hidden_states.shape[0] > 0:
                hidden_states, scales = torch_npu.npu_dynamic_mx_quant(
                    hidden_states, dst_type=torch.float8_e4m3fn
                )
                scales = normalize_mxfp_scale(scales).view(torch.uint8)
            else:
                hidden_states = hidden_states.new_empty(
                    (0, self.hidden_size), dtype=torch.float8_e4m3fn
                )
                scales = hidden_states.new_empty(
                    (0, (self.hidden_size + 63) // 64, 2), dtype=torch.uint8
                )

        state = _AllGatherState(
            group=group,
            row_sizes=row_sizes,
            input_dtype=input_dtype,
        )
        return AscendOpsDispatchOutput(
            hidden_states=gather_rows(hidden_states),
            hidden_states_scale=gather_rows(scales) if scales is not None else None,
            topk_weights=gather_rows(topk_weights.float()),
            topk_ids=gather_rows(topk_ids.to(torch.int32)),
            expert_tokens=None,
            group_list_type=1,
            exec_mode="allgather",
            quant_mode=quant_mode,
            forward_state=state,
        )

    def _combine_all_gather(
        self, contributions: torch.Tensor, state: _AllGatherState
    ) -> torch.Tensor:
        """ReduceScatter contributions back to the local shard."""
        sizes = state.row_sizes
        group = state.group
        max_rows = max(sizes)
        if min(sizes) == max_rows:
            padded = contributions
        else:
            padded = contributions.new_zeros((len(sizes) * max_rows, self.hidden_size))
            offset = 0
            for rank, rows in enumerate(sizes):
                self._pad_rows(
                    contributions[offset : offset + rows],
                    max_rows,
                    out=padded[rank * max_rows : (rank + 1) * max_rows],
                )
                offset += rows
        output = contributions.new_empty((max_rows, self.hidden_size))
        dist.reduce_scatter_tensor(output, padded, group=group)
        output = output[: sizes[dist.get_rank(group)]].to(state.input_dtype)
        return output

    # Prefill: AllToAllV

    def _resolve_all_to_all_counts(
        self, topk_ids: torch.Tensor, group
    ) -> tuple[list[int], list[int], torch.Tensor]:
        if self.num_experts is None or self.num_local_experts is None:
            raise ValueError("Ascend Ops alltoallv requires expert counts")
        ep_size = dist.get_world_size(group=group)
        if self.num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by EP size={ep_size}"
            )
        local_experts = self.num_experts // ep_size
        if self.num_local_experts != local_experts:
            raise ValueError(
                "Ascend Ops alltoallv requires num_local_experts to match "
                f"num_experts / ep_size ({self.num_local_experts} != {local_experts})"
            )

        if topk_ids.numel() == 0:
            local_counts = torch.zeros(
                self.num_experts, dtype=torch.int64, device=topk_ids.device
            )
        else:
            flat_topk_ids = topk_ids.reshape(-1)
            valid_topk_ids = flat_topk_ids[
                (flat_topk_ids >= 0) & (flat_topk_ids < self.num_experts)
            ]
            if valid_topk_ids.numel() == 0:
                local_counts = torch.zeros(
                    self.num_experts, dtype=torch.int64, device=topk_ids.device
                )
            else:
                local_counts = torch.histc(
                    valid_topk_ids.to(torch.float32),
                    bins=self.num_experts,
                    min=0,
                    max=self.num_experts,
                ).to(torch.int64)
        rank = dist.get_rank(group=group)
        if ep_size == 1:
            gathered_counts = local_counts.unsqueeze(0)
        else:
            count_chunk = local_counts.to(torch.int32)
            gathered_flat = torch.empty(
                ep_size * local_experts,
                dtype=torch.int32,
                device=local_counts.device,
            )
            split = [local_experts] * ep_size
            dist.all_to_all_single(
                gathered_flat,
                count_chunk,
                output_split_sizes=split,
                input_split_sizes=split,
                group=group,
            )
            gathered_counts = gathered_flat.view(ep_size, local_experts).to(
                torch.int64
            )
        local_start = rank * local_experts
        local_end = local_start + local_experts
        recv_counts = (
            gathered_counts
            if ep_size > 1
            else gathered_counts[:, local_start:local_end]
        )
        input_splits = (
            local_counts.view(ep_size, local_experts).sum(dim=1).cpu().tolist()
        )
        output_splits = recv_counts.sum(dim=1).cpu().tolist()
        return input_splits, output_splits, recv_counts

    def _all_to_all_single(
        self,
        hidden_states: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        group,
    ) -> torch.Tensor:
        rank = dist.get_rank(group=group)
        ep_size = dist.get_world_size(group=group)
        total_output = int(sum(output_splits))
        total_input = int(sum(input_splits))
        if hidden_states.shape[0] != total_input:
            raise RuntimeError(
                "Ascend AllToAllV input split mismatch: "
                f"tensor has {hidden_states.shape[0]} rows, "
                f"but input_splits sum to {total_input}"
            )

        # HCCL all_to_all_single can hang or no-op when either side of a rank
        # has a zero-size split.  This can still happen in the reverse
        # combine exchange when a rank owns no routed expert rows, even after
        # the dispatch input was padded before routing.  Add a self-loop
        # dummy element and remove exactly that received row afterwards.
        padded = False
        dummy_output_pos = 0
        if ep_size > 1 and (total_input == 0 or total_output == 0):
            padded = True
            orig_input_splits = list(input_splits)
            orig_output_splits = list(output_splits)
            output_splits = list(output_splits)
            input_splits = list(input_splits)
            output_splits[rank] += 1
            input_splits[rank] += 1
            total_output += 1
            # Insert the self-loop row at the beginning of this rank's send
            # segment so all pre-existing split boundaries remain intact.
            insert_pos = sum(orig_input_splits[:rank])
            dummy_output_pos = sum(orig_output_splits[:rank])
            dummy = hidden_states.new_zeros(
                (1,) + tuple(hidden_states.shape[1:])
            )
            hidden_states = torch.cat(
                [hidden_states[:insert_pos], dummy, hidden_states[insert_pos:]],
                dim=0,
            )

        output = hidden_states.new_empty(
            (max(total_output, 0),) + tuple(hidden_states.shape[1:])
        )
        dist.all_to_all_single(
            output,
            hidden_states.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )

        if padded:
            output = torch.cat(
                [
                    output[:dummy_output_pos],
                    output[dummy_output_pos + 1 :],
                ],
                dim=0,
            )

        return output

    def _dispatch_all_to_all_v(self, hidden_states, topk_weights, topk_ids, group):
        """Send each token to its expert-owning rank."""
        original_hidden_shape = hidden_states.shape
        original_rows = hidden_states.shape[0]
        topk_ids = topk_ids.to(torch.int32)
        # Keep empty DP/TP shards out of the Ascend routing kernels.  In
        # particular, npu_moe_init_routing_v2 and the following grouped
        # matmul paths are not valid with a zero-row input.  vLLM-Ascend
        # applies the same rule before token permutation: every shard gets at
        # least one route, and the padded row is removed after combine.
        if original_rows == 0:
            hidden_states = hidden_states.new_zeros(
                (1,) + tuple(hidden_states.shape[1:])
            )
            topk_ids = topk_ids.new_zeros(
                (1,) + tuple(topk_ids.shape[1:])
            )
            if topk_weights is not None:
                topk_weights = topk_weights.new_zeros(
                    (1,) + tuple(topk_weights.shape[1:])
                )
        input_splits, output_splits, recv_counts = self._resolve_all_to_all_counts(
            topk_ids, group
        )
        quant_mode = self._resolve_quant_mode("routing")
        routed = npu_moe_init_routing(
            hidden_states, topk_ids, self.num_experts, quant_mode=quant_mode
        )

        scale_for_comm = routed.scale
        if quant_mode not in (-1, INT8_QUANT_MODE) and scale_for_comm is not None:
            scale_for_comm = scale_for_comm.view(torch.uint8)
        received_scale = (
            self._all_to_all_single(scale_for_comm, output_splits, input_splits, group)
            if scale_for_comm is not None
            else None
        )
        received = self._all_to_all_single(
            routed.hidden_states, output_splits, input_splits, group
        )
        state = _AllToAllVState(
            group=group,
            input_splits=input_splits,
            output_splits=output_splits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expanded_row_idx=routed.expanded_row_idx.clone(),
            hidden_shape=original_hidden_shape,
            padded_hidden_shape=hidden_states.shape,
        )
        return AscendOpsDispatchOutput(
            hidden_states=received,
            hidden_states_scale=received_scale,
            topk_weights=None,
            topk_ids=None,
            expert_tokens=None,
            group_list_type=1,
            exec_mode="alltoallv",
            quant_mode=quant_mode,
            recv_counts=recv_counts,
            forward_state=state,
        )

    def _combine_all_to_all_v(
        self, local_output: torch.Tensor, state: _AllToAllVState
    ) -> torch.Tensor:
        """Reverse AllToAllV and apply source-side router weights."""
        source_order = self._all_to_all_single(
            local_output,
            state.input_splits,
            state.output_splits,
            state.group,
        )
        if source_order.shape[0] == 0:
            if state.hidden_shape[0] != 0:
                raise RuntimeError(
                    "Ascend AllToAllV combine returned no rows for a non-empty "
                    f"source batch of {state.hidden_shape[0]} rows"
                )
            output = source_order.new_empty(state.hidden_shape)
        else:
            output = npu_moe_finalize_routing(
                source_order,
                state.expanded_row_idx,
                topk_weights=state.topk_weights,
            ).view(state.padded_hidden_shape)
            output = output[: state.hidden_shape[0]]
        return output
