"""CPU coverage of the V4.1 DSpark PD configuration gate."""

import enum
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from test_dsv41_a3_cache import ROOT, _load_definitions


class VerifyMode(enum.Enum):
    STATIC = "static"
    COMPACT = "compact"
    CAP_ACCEPT = "cap-accept"


def validate(*, mode=VerifyMode.STATIC, **overrides):
    cfg = SimpleNamespace(
        device="npu",
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="ascend",
        speculative_algorithm="DSPARK",
        dsv4_attn_backend="auto",
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=False,
        enable_hisparse=False,
        enable_two_batch_overlap=False,
        pp_size=1,
        dcp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        cp_strategy="interleave",
        tp_size=16,
        attn_dp_size=8,
        moe_a2a_backend="deepep",
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="disabled")),
    )
    vars(cfg).update(overrides)
    gate = ModuleType("sglang.srt.layers.attention.dsv4.unified_kv_gate")
    gate.is_unified_kv_triton = lambda: False
    ragged = ModuleType("sglang.srt.speculative.ragged_verify")
    ragged.RaggedVerifyMode = VerifyMode
    ragged.read_ragged_verify_mode = lambda: mode
    ns = {
        "resolving_view": lambda _: cfg,
        "model_config_of": lambda _: SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        ),
        "Backend": SimpleNamespace(DISABLED="disabled", BREAKABLE="breakable"),
        "attn_dp_enabled_of": lambda c: c.attn_dp_size > 1,
        "get_platform": lambda: SimpleNamespace(
            is_cuda=cfg.device == "cuda", is_hip=False
        ),
    }
    _load_definitions(
        ROOT / "python/sglang/srt/arg_groups/deepseek_v4_hook.py",
        {"validate_deepseek_v41_features"},
        ns,
    )
    with patch.dict(sys.modules, {gate.__name__: gate, ragged.__name__: ragged}):
        ns["validate_deepseek_v41_features"](cfg)


class TestAscendDSparkPDConfig(unittest.TestCase):
    def test_tp16_dp8_both_roles(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                validate(disaggregation_mode=role)

    def test_ascend_requires_npu(self):
        with self.assertRaisesRegex(ValueError, "device npu"):
            validate(device="cuda")

    def test_ascend_disallows_prefill_cp_even_when_implicit(self):
        for role in ("prefill", "decode"):
            for overrides in ({"enable_prefill_cp": True}, {"attn_cp_size": 2}):
                with self.subTest(role=role, **overrides):
                    with self.assertRaisesRegex(ValueError, "CP=1 on both"):
                        validate(disaggregation_mode=role, **overrides)

    def test_static_verify_remains_required(self):
        for mode in (VerifyMode.COMPACT, VerifyMode.CAP_ACCEPT):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "static verify"):
                    validate(mode=mode)

    def test_dcp_and_pp_remain_disallowed(self):
        with self.assertRaisesRegex(ValueError, "DCP=1"):
            validate(dcp_size=2)
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            validate(pp_size=2)

    def test_other_transfer_backends_and_gpu_prefill_cp(self):
        for backend in ("mooncake", "nixl", "mori"):
            with self.subTest(backend=backend):
                validate(device="cuda", disaggregation_transfer_backend=backend)
                validate(
                    device="cuda",
                    disaggregation_transfer_backend=backend,
                    enable_prefill_cp=True,
                    attn_cp_size=2,
                )
                with self.assertRaisesRegex(ValueError, "CP=1 on decode"):
                    validate(
                        device="cuda",
                        disaggregation_transfer_backend=backend,
                        disaggregation_mode="decode",
                        attn_cp_size=2,
                    )

    def test_unknown_transport_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "DSpark PD requires"):
            validate(disaggregation_transfer_backend="fake")

    def test_non_speculative_pd_unchanged(self):
        validate(speculative_algorithm=None)


if __name__ == "__main__":
    unittest.main()
