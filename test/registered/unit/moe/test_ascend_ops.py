"""CPU unit tests for the Ascend MoE dispatcher contracts.

These tests deliberately stop at the Python dispatcher boundary.  The
``torch_npu`` kernels themselves require an Ascend runner and are covered by
hardware integration tests when an NPU job is available.
"""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.moe.mega_moe import (
    get_mega_moe_sym_buffer,
    mega_moe_activation_args,
    mega_moe_quant_args,
    mega_moe_weight_list,
    run_mega_moe,
)
from sglang.srt.layers.moe.token_dispatcher.ascend_ops import (
    AscendOpsCombineInput,
    AscendOpsDispatcher,
    _AllGatherState,
    _AllToAllVState,
    _MegaMoEState,
    _MC2State,
)
from sglang.srt.layers.moe.utils import AscendOpsConfig, DispatcherOutputDtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAscendOpsConfig(CustomTestCase):
    # Protects the configuration boundary for the MegaMoE fused decode path.
    def test_fused_op_is_accepted_during_config_validation(self):
        config = AscendOpsConfig.from_json_str('{"decode":"fused_op"}')
        self.assertEqual(config.decode, "fused_op")

    # Protects the default path used when no Ascend Ops config is supplied.
    def test_low_latency_is_the_default_decode_mode(self):
        config = AscendOpsConfig.from_json_str(None)
        self.assertEqual(config.decode, "low_latency")


class TestAscendOpsDispatcherContracts(CustomTestCase):
    def test_mega_moe_retries_uninitialized_ep_comm_without_caching(self):
        """An unavailable EP communicator must not publish a reusable buffer."""
        module = "sglang.srt.hardware_backend.npu.moe.mega_moe"
        cache = {}
        ready = {"reduced": False, "synced": False}
        backend = Mock()
        backend.get_hccl_comm_name.side_effect = ["", "hccl_ep_comm"]
        group = Mock()
        group._get_backend.return_value = backend
        buffer = object()

        def all_reduce(probe, *, group):
            ready["reduced"] = True

        def synchronize(device):
            self.assertTrue(ready["reduced"])
            ready["synced"] = True

        def create_buffer(actual_group, **kwargs):
            self.assertIs(actual_group, group)
            self.assertTrue(ready["synced"])
            return buffer

        provider = ModuleType("sgl_kernel_npu.modules.moe")
        provider.get_symm_buffer_for_mega_moe = Mock(side_effect=create_buffer)
        config = SimpleNamespace(
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size_per_partition=2048,
        )
        cpu_device = torch.device("cpu")
        probe = torch.ones(1)
        npu = SimpleNamespace(current_device=lambda: 0, synchronize=synchronize)
        with (
            patch.dict("sys.modules", {provider.__name__: provider}),
            patch(f"{module}._SYMM_BUFFER_CACHE", cache),
            patch(f"{module}.torch.npu", npu, create=True),
            patch(f"{module}.torch.device", return_value=cpu_device),
            patch(f"{module}.torch.ones", return_value=probe),
            patch(f"{module}.torch.distributed.all_reduce", side_effect=all_reduce),
            patch(f"{module}.torch.distributed.get_rank", return_value=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "comm name is empty"):
                get_mega_moe_sym_buffer(config, {}, group, 3, "normal")
            self.assertEqual(cache, {})
            self.assertEqual(provider.get_symm_buffer_for_mega_moe.call_count, 0)

            self.assertIs(
                get_mega_moe_sym_buffer(config, {}, group, 3, "normal"), buffer
            )
            ready.update(reduced=False, synced=False)
            self.assertIs(
                get_mega_moe_sym_buffer(config, {}, group, 2, "normal"), buffer
            )
            self.assertEqual(ready, {"reduced": False, "synced": False})
            self.assertEqual(provider.get_symm_buffer_for_mega_moe.call_count, 1)

    def _dispatcher_for_quant_mode(self, quant_config=None):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        dispatcher.quant_config = quant_config or {}
        return dispatcher

    # Protects mode-specific quantization settings from being ignored in favor
    # of the shared dispatcher dtype.
    def test_mode_specific_dtype_overrides_shared_dtype(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {
                "dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "mxfp8",
            }
        )
        self.assertEqual(
            dispatcher._resolve_quant_mode("mc2", "low_latency"),
            4,
        )
        self.assertEqual(dispatcher._resolve_quant_mode("routing"), -1)

    # Protects MC2 from receiving a routing-only dispatcher dtype it cannot
    # encode in its own quantization enum.
    def test_unsupported_mc2_dtype_is_rejected(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {"dispatcher_output_dtype": DispatcherOutputDtype.FP8.value}
        )
        with self.assertRaisesRegex(NotImplementedError, "mc2.*fp8"):
            dispatcher._resolve_quant_mode("mc2")

    # Protects the MegaMoE interface from using the MC2 quant enum for routing.
    def test_mega_moe_quant_args_follow_dispatch_dtype(self):
        self.assertEqual(
            mega_moe_quant_args({"dispatcher_output_dtype": "bf16"}), (0, None)
        )
        self.assertEqual(
            mega_moe_quant_args({"dispatcher_output_dtype": "int8"}),
            (2, torch.int8),
        )
        mode, dtype = mega_moe_quant_args({"dispatcher_output_dtype": "mxfp8"})
        self.assertEqual(mode, 4)
        self.assertEqual(dtype, torch.float8_e4m3fn)
        self.assertEqual(
            mega_moe_quant_args(
                {
                    "normal_dispatcher_output_dtype": "bf16",
                    "low_latency_dispatcher_output_dtype": "mxfp8",
                },
                "normal",
            ),
            (0, None),
        )

    # Protects model activation settings from being silently changed by fused dispatch.
    def test_mega_moe_activation_args_preserve_silu_clamp(self):
        config = type(
            "Config",
            (),
            {
                "activation": "silu",
                "gemm1_clamp_limit": 7.0,
                "gemm1_alpha": None,
                "swiglu_limit": None,
            },
        )()
        self.assertEqual(mega_moe_activation_args(config), ("swiglustep", 7.0, None))

    # Protects stacked expert weights from being passed as one expert tensor.
    def test_mega_moe_weight_list_unstacks_expert_dimension(self):
        weights = torch.zeros((2, 3, 4))
        result = mega_moe_weight_list(weights)
        self.assertEqual(len(result), 2)
        self.assertEqual(tuple(result[0].shape), (3, 4))

    # Protects four-dimensional MXFP expert scales from being passed as one
    # packed list item instead of one item per expert.
    def test_mega_moe_weight_list_unstacks_packed_scales(self):
        scales = torch.zeros((2, 3, 4, 2))
        result = mega_moe_weight_list(scales, expected_len=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(tuple(result[0].shape), (3, 4, 2))

    # Protects the MegaMoE call boundary and the argument layout passed to the
    # registered sgl-kernel-npu operator.
    def test_mega_moe_dispatch_calls_registered_operator(self):
        mega_moe = Mock(return_value=("output", "aux"))
        fake_moe_module = ModuleType("sgl_kernel_npu.modules.moe")
        fake_moe_module.mega_moe = mega_moe
        fake_parent = ModuleType("sgl_kernel_npu.modules")
        fake_parent.moe = fake_moe_module
        fake_root = ModuleType("sgl_kernel_npu")
        fake_root.modules = fake_parent

        runner_config = SimpleNamespace(
            num_experts=2,
            top_k=2,
            hidden_size=4,
            intermediate_size_per_partition=3,
            layer=SimpleNamespace(
                w13_weight=torch.zeros((2, 3, 4)),
                w2_weight=torch.zeros((2, 4, 3)),
                w13_weight_scale=torch.zeros((2, 3, 4, 2)),
                w2_weight_scale=torch.zeros((2, 4, 3, 2)),
            ),
            activation="silu",
            gemm1_clamp_limit=None,
            gemm1_alpha=None,
            swiglu_limit=None,
        )
        hidden_states = torch.zeros((1, 4))
        topk_weights = torch.ones((1, 2))
        topk_ids = torch.zeros((1, 2), dtype=torch.int32)

        with patch.dict(
            "sys.modules",
            {
                "sgl_kernel_npu": fake_root,
                "sgl_kernel_npu.modules": fake_parent,
                "sgl_kernel_npu.modules.moe": fake_moe_module,
            },
        ), patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        ), patch(
            "sglang.srt.hardware_backend.npu.moe.mega_moe.check_mega_moe_kernel",
        ), patch(
            "sglang.srt.hardware_backend.npu.moe.mega_moe.get_mega_moe_sym_buffer",
            return_value="sym_buffer",
        ), patch(
            "sglang.srt.runtime_context.get_parallel",
        ):
            result = run_mega_moe(
                runner_config,
                {"dispatcher_output_dtype": "bf16"},
                hidden_states,
                topk_weights,
                topk_ids,
            )

        self.assertEqual(result, "output")
        args, kwargs = mega_moe.call_args
        self.assertIs(args[0], hidden_states)
        self.assertIs(args[1], topk_ids)
        self.assertEqual(args[2].dtype, torch.float32)
        self.assertEqual(len(args[3]), 2)
        self.assertEqual(len(args[4]), 2)
        self.assertEqual(args[5], "sym_buffer")
        self.assertEqual(len(kwargs["l1_weights_sf"]), 2)
        self.assertEqual(len(kwargs["l2_weights_sf"]), 2)
        self.assertEqual(kwargs["activation"], "swiglu")
        self.assertNotIn("activation_params", kwargs)

    # Protects MegaMoE's forward state from being confused with communication
    # states when combine receives the runner output.
    def test_mega_moe_forward_state_is_a_valid_combine_state(self):
        dispatcher = self._dispatcher_for_quant_mode()
        hidden_states = torch.zeros((1, 4))
        state = _MegaMoEState(
            topk_weights=torch.ones((1, 2)),
            topk_ids=torch.zeros((1, 2), dtype=torch.int32),
        )

        with patch.object(
            dispatcher, "_combine_megamoe", return_value=hidden_states
        ) as combine:
            result = dispatcher.combine(AscendOpsCombineInput(hidden_states, state))

        self.assertIs(result, hidden_states)
        combine.assert_called_once_with(hidden_states, state)

    # Protects dispatch from executing MegaMoE before the normal runner hooks
    # have received the forward state.
    def test_mega_moe_dispatch_builds_forward_state(self):
        dispatcher = self._dispatcher_for_quant_mode(
            {"dispatcher_output_dtype": "bf16"}
        )
        hidden_states = torch.zeros((1, 4))
        topk_weights = torch.ones((1, 2))
        topk_ids = torch.zeros((1, 2), dtype=torch.int64)

        output = dispatcher._dispatch_fused_op(
            hidden_states, topk_weights, topk_ids
        )

        self.assertEqual(output.exec_mode, "fused_op")
        self.assertIsInstance(output.forward_state, _MegaMoEState)
        self.assertIs(output.forward_state.topk_weights, output.topk_weights)
        self.assertEqual(output.topk_weights.dtype, torch.float32)
        self.assertEqual(output.forward_state.topk_ids.dtype, torch.int32)
        self.assertEqual(output.forward_state.dtype_mode, "low_latency")

    # Protects AllGather row metadata when attention CP/DP creates uneven shards.
    def test_all_gather_row_sizes_use_actual_rank_local_rows(self):
        def fake_all_gather(outputs, local_rows, group):
            outputs[0].copy_(local_rows)
            outputs[1].fill_(3)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.all_gather",
                side_effect=fake_all_gather,
            ),
        ):
            row_sizes = AscendOpsDispatcher._all_gather_row_sizes(
                2, group=None, device=torch.device("cpu")
            )

        self.assertEqual(row_sizes, [2, 3])

    # Protects the DP+attention-TP all-gather fast path from adding a device
    # row-count collective when scheduler metadata already determines rows.
    def test_all_gather_row_sizes_use_dp_metadata_without_collective(self):
        dispatcher = AscendOpsDispatcher.__new__(AscendOpsDispatcher)
        parallel = SimpleNamespace(attn_dp_size=2, attn_cp_size=1, attn_tp_size=2)
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_dp_global_num_tokens",
                return_value=[8, 4],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
        ):
            layout = AscendOpsDispatcher._resolve_row_layout(
                dispatcher, 4, group=None, device=torch.device("cpu"), allow_cp=True
            )

        self.assertEqual(layout.row_sizes, (4, 4, 2, 2))

    # Protects the metadata fast path from silently falling back to a device
    # collective when the upstream attention layout is inconsistent.
    def test_allgather_metadata_row_mismatch_is_rejected(self):
        parallel = SimpleNamespace(attn_dp_size=2, attn_cp_size=1, attn_tp_size=2)
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_dp_global_num_tokens",
                return_value=[8, 4],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "local rows disagree"):
                AscendOpsDispatcher._resolve_row_layout(
                    AscendOpsDispatcher.__new__(AscendOpsDispatcher),
                    3,
                    group=None,
                    device=torch.device("cpu"),
                    allow_cp=True,
                )

    # Protects MegaMoE capacity from using the unsplit DP token count when
    # attention TP has already divided the local MoE rows.
    def test_fused_op_pads_to_attention_tp_local_capacity(self):
        dispatcher = self._dispatcher_for_quant_mode()
        hidden_states = torch.zeros((2, 4))
        topk_weights = torch.ones((2, 1))
        topk_ids = torch.zeros((2, 1), dtype=torch.int64)
        parallel = SimpleNamespace(
            attn_dp_rank=1,
            attn_dp_size=2,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_cp_size=1,
        )
        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_dp_global_num_tokens",
                return_value=[8, 4],
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops."
                "get_is_extend_in_batch",
                return_value=True,
            ),
        ):
            output = dispatcher._dispatch_fused_op(
                hidden_states, topk_weights, topk_ids
            )

        self.assertEqual(output.hidden_states.shape[0], 4)
        self.assertEqual(output.forward_state.num_valid_rows, 2)

    # Protects alltoallv count exchange from all-gathering the full global
    # histogram when each rank only needs its local-expert slice.
    def test_alltoallv_counts_exchange_local_expert_slices(self):
        dispatcher = self._dispatcher_for_quant_mode()
        dispatcher.num_experts = 4
        dispatcher.num_local_experts = 2
        topk_ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.int32)

        def fake_all_to_all(output, input, **kwargs):
            output.copy_(input)

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_world_size",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist.get_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.ascend_ops.dist."
                "all_to_all_single",
                side_effect=fake_all_to_all,
            ) as all_to_all,
        ):
            input_splits, output_splits, recv_counts = dispatcher._resolve_all_to_all_counts(
                topk_ids, group=None
            )

        self.assertEqual(input_splits, [2, 2])
        self.assertEqual(output_splits, [2, 2])
        self.assertEqual(tuple(recv_counts.shape), (2, 2))
        all_to_all.assert_called_once()

    # Protects the forward-state handoff: combine must work without a cached
    # dispatch result stored on the dispatcher instance.
    def test_combine_uses_forward_state_instead_of_dispatcher_state(self):
        dispatcher = self._dispatcher_for_quant_mode()
        state = _MC2State(
            num_tokens=1,
            assist_info_for_combine=torch.empty(0),
            ep_send_counts=torch.empty(0),
            x_active_mask=None,
            expand_scales=None,
            topk_ids=torch.zeros((1, 1), dtype=torch.int32),
            topk_weights=torch.ones((1, 1)),
        )
        combine_input = AscendOpsCombineInput(
            hidden_states=torch.zeros((1, 4)),
            forward_state=state,
        )
        with patch.object(
            dispatcher, "_combine_mc2", return_value="combined"
        ) as combine:
            result = dispatcher.combine(combine_input)

        self.assertEqual(result, "combined")
        combine.assert_called_once_with(combine_input.hidden_states, state)

    # Protects the explicit state-to-combine mapping for both prefill paths.
    def test_combine_dispatches_all_prefill_state_types(self):
        dispatcher = self._dispatcher_for_quant_mode()
        hidden_states = torch.zeros((1, 4))
        states = (
            (_AllGatherState(None, [1], torch.bfloat16), "allgather"),
            (
                _AllToAllVState(
                    None,
                    [1],
                    [1],
                    torch.ones((1, 1)),
                    torch.zeros((1, 1), dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                    torch.Size((1, 4)),
                ),
                "alltoallv",
            ),
        )
        for state, mode in states:
            with self.subTest(mode=mode):
                combine_name = f"_combine_{mode}"
                with patch.object(
                    dispatcher, combine_name, return_value=mode
                ) as combine:
                    result = dispatcher.combine(
                        AscendOpsCombineInput(hidden_states, state)
                    )
                self.assertEqual(result, mode)
                combine.assert_called_once_with(hidden_states, state)


if __name__ == "__main__":
    unittest.main()
