"""MegaMoE execution adapter for the Ascend MoE runner path."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs


_SYMM_BUFFER_CACHE: dict[tuple, tuple[int, object]] = {}

_MEGA_MOE_SUPPORTED_KERNELS_MSG = (
    "ascend_ops fused_op supports NPUUnquantMoEMethod, "
    "NPUW8A8Int8MoEMethod, NPUW4A8Int8MoEMethod, NPUMXFP8MoEMethod, "
    "and NPUW4A8MXFP4MoEMethod"
)


def mega_moe_quant_args(
    quant_config: Optional[dict], dtype_mode: Optional[str] = None
) -> Tuple[int, Any]:
    """Map the configured dispatcher dtype to MegaMoE's quantization pair."""

    config = quant_config or {}
    dtype = (
        config.get(f"{dtype_mode}_dispatcher_output_dtype")
        if dtype_mode is not None
        else config.get("low_latency_dispatcher_output_dtype")
    )
    if dtype is None:
        dtype = config.get("dispatcher_output_dtype")
    dtype = getattr(dtype, "value", dtype)
    if dtype in (None, "bf16", "bfloat16"):
        return 0, None
    if dtype == "int8":
        # A2/A3 CANN tiling validates the output dtype independently from the
        # quant mode.  Leaving this as None makes the provider use its default
        # dtype code (BF16/0), which is invalid for INT8 quant mode 2.
        return 2, torch.int8
    if dtype in ("mxfp8", "mx_fp8_e4m3", "float8_e4m3fn"):
        return 4, torch.float8_e4m3fn
    if dtype in ("mxfp4", "mx_fp4_e2m1"):
        try:
            import torch_npu

            return 4, torch_npu.float4_e2m1fn_x2
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("MegaMoE MXFP4 requires torch_npu FP4 dtype") from exc
    raise ValueError(f"Unsupported MegaMoE dispatcher dtype: {dtype!r}")


def mega_moe_activation_args(
    config,
) -> Tuple[str, Optional[float], Optional[Dict[str, float]]]:
    """Translate ``MoeRunnerConfig`` activation fields for MegaMoE."""

    activation = config.activation or "silu"
    if activation == "situ":
        params = {"beta": config.gemm1_alpha or 4.0}
        if config.gemm1_clamp_limit is not None:
            params["linear_beta"] = config.gemm1_clamp_limit
        return "situglu", None, params
    if activation == "npu_swiglu_oai":
        return "swigluoai", config.gemm1_clamp_limit, {
            "alpha": config.gemm1_alpha or 1.702,
            "beta": 1.0,
        }
    if activation == "gelu":
        return "gelu", None, None
    if config.gemm1_clamp_limit is not None:
        return "swiglustep", config.gemm1_clamp_limit, None
    if config.swiglu_limit is not None and config.swiglu_limit > 0:
        return "swiglu", float(config.swiglu_limit), None
    return "swiglu", None, None


def mega_moe_weight_list(weight, expected_len: Optional[int] = None) -> list:
    """Return the per-expert list required by the public operator.

    Expert scales can be four-dimensional after MXFP layout conversion, while
    int8 scales can be two-dimensional after squeezing their trailing axis.
    The first dimension is still the expert dimension in both cases.  Use the
    expected expert count when the tensor rank alone is not enough to identify
    that layout.
    """

    if isinstance(weight, (list, tuple)):
        values = list(weight)
        if (
            expected_len is not None
            and len(values) == 1
            and isinstance(values[0], torch.Tensor)
            and values[0].ndim > 0
            and values[0].shape[0] == expected_len
        ):
            return list(values[0].unbind(0))
        return values
    if (
        isinstance(weight, torch.Tensor)
        and expected_len is not None
        and weight.ndim > 0
        and weight.shape[0] == expected_len
    ):
        return list(weight.unbind(0))
    if isinstance(weight, torch.Tensor) and weight.dim() == 3:
        return list(weight.unbind(0))
    return [weight]


def mega_moe_bias_list(layer, prefix: str, expected_len: int):
    """Return the compensation bias used by the selected NPU quantizer.

    W4A8 ModelSlim quantization stores the MegaMoE compensation term as
    ``*_scale_bias`` after weight processing.  Other quantizers expose the
    same operator input as ``*_weight_bias``.  The two names are alternative
    representations of the same MegaMoE argument, so prefer the processed
    scale bias and fall back to the conventional weight bias.
    """

    for suffix in ("scale_bias", "weight_bias"):
        value = getattr(layer, f"{prefix}_{suffix}", None)
        if value is not None:
            return mega_moe_weight_list(value, expected_len)
    return None


def _fp32_bits_to_uint64(scale: torch.Tensor) -> torch.Tensor:
    """Reference ``pack_scale_to_uint64``: fp32 bits widened into UINT64."""

    return (
        scale.reshape(-1).contiguous().view(torch.int32).to(torch.int64).view(torch.uint64)
    )


def _pack_mega_moe_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert one processed NPU scale tensor to the A3 MegaMoE layout."""

    scale = scale.contiguous()
    if scale.dtype in (torch.float16, torch.bfloat16):
        # The INT8 dispatch contract takes UINT64 scales whose low 32 bits hold
        # the fp32 value, so the per-channel half-precision scale the W8A8
        # quantizer keeps has to be widened back instead of passed through.
        return _fp32_bits_to_uint64(scale.float())
    if scale.dtype == torch.float32:
        scale_fp32 = scale.reshape(-1)
        if scale_fp32.numel() % 2:
            raise ValueError(
                "MegaMoE FP scales must contain an even number of values for "
                f"UINT64 packing, got shape {tuple(scale.shape)}"
            )
        return (
            scale_fp32.view(torch.int32).reshape(-1, 2).view(torch.uint64).reshape(-1)
        )
    if scale.dtype in (torch.int64, torch.uint64):
        return scale.view(torch.int64).reshape(-1).view(torch.uint64)
    return scale.reshape(-1)


def mega_moe_uses_mxfp_scales(layer) -> bool:
    """Whether the layer keeps E8M0 block scales instead of packed INT64."""

    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUMXFP8MoEMethod,
        NPUW4A8MXFP4MoEMethod,
    )

    return isinstance(
        getattr(layer, "w2_kernel", None),
        (NPUMXFP8MoEMethod, NPUW4A8MXFP4MoEMethod),
    )


def mega_moe_scale_list(
    layer, name: str, expected_len: int, *, mxfp: bool = False
):
    """Return per-expert scales in the format required by MegaMoE.

    The normal Ascend grouped matmul path keeps scales in a stacked tensor and
    uses BF16 or packed INT64 values.  MegaMoE receives one flattened scale
    tensor per expert and interprets the packed values as UINT64.
    """

    if mxfp:
        stacked = getattr(layer, f"{name}_mega_moe", None)
        if stacked is not None:
            # Kept at load time as the operator's [E, rows, K/64, 2] ND layout;
            # the shared GMM layout would be transposed and unusable here.
            return [stacked]
    value = getattr(layer, name, None)
    if value is None:
        # MXFP checkpoints store the block scale under the ``_inv`` name; the
        # shared A5 GMM path passes the same tensor as its weight scale.
        value = getattr(layer, f"{name}_inv", None)
    if value is None:
        return None
    items = list(mega_moe_weight_list(value, expected_len))
    if mxfp:
        # A5 keeps E8M0 block scales as [rows, K/64, 2]; the operator wants the
        # typed view, not the packed uint64 form used by the INT paths.
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is not None:
            items = [item.view(e8m0) if item.dtype == torch.uint8 else item for item in items]
        return items
    return [_pack_mega_moe_scale(item) for item in items]


def mega_moe_weight_type(layer):
    """FP4 weight dtype the MXFP4-weight providers require.

    The packed-FP4 quantizer stores each weight as two 4-bit values per byte;
    the operator cannot infer that from the tensor shape, so it needs an
    explicit weight type. Other quantizers pass int4 in int32 and must leave
    the parameter unset.
    """

    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUW4A8MXFP4MoEMethod,
    )

    if not isinstance(getattr(layer, "w2_kernel", None), NPUW4A8MXFP4MoEMethod):
        return None
    import torch_npu

    dtype = getattr(torch_npu, "float4_e2m1fn_x2", None)
    if dtype is None:
        raise RuntimeError("MXFP4 MegaMoE requires torch_npu.float4_e2m1fn_x2")
    return dtype


def check_mega_moe_kernel(layer) -> None:
    """Reject quantization methods that the operator interface cannot serve."""

    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUMXFP8MoEMethod,
        NPUUnquantMoEMethod,
        NPUW4A8Int8MoEMethod,
        NPUW4A8MXFP4MoEMethod,
        NPUW8A8Int8MoEMethod,
    )

    supported = (
        NPUUnquantMoEMethod,
        NPUW8A8Int8MoEMethod,
        NPUW4A8Int8MoEMethod,
        NPUMXFP8MoEMethod,
        NPUW4A8MXFP4MoEMethod,
    )
    kernel = getattr(layer, "w2_kernel", None)
    if not isinstance(kernel, supported):
        raise NotImplementedError(
            f"unsupported MegaMoE quant method {type(kernel).__name__}; "
            f"{_MEGA_MOE_SUPPORTED_KERNELS_MSG}"
        )


def _symm_buffer_key(runner_config, quant_config, group, dtype_mode) -> tuple:
    """Cache key of the SymmBuffer shared by one runner configuration."""

    quant_mode, quant_dtype = mega_moe_quant_args(quant_config, dtype_mode)
    return (
        id(group),
        int(runner_config.num_experts),
        int(runner_config.top_k),
        int(runner_config.hidden_size),
        int(runner_config.intermediate_size_per_partition),
        quant_mode,
        quant_dtype,
    )


def _padded_capacity(num_tokens: int) -> Optional[int]:
    """Fixed capacity the batch is padded to, or None when padding is off.

    A3 MegaMoE expects every rank to enter with the same capacity and marks the
    real rows with ``x_active_mask``; the value therefore has to come from a
    configuration every rank shares, never from local history.
    """

    if not envs.SGLANG_MEGAMOE_PAD_TO_CAPACITY.get():
        return None
    capacity = envs.SGLANG_MEGAMOE_MAX_TOKENS_PER_RANK.get()
    if capacity <= 0:
        return None
    if num_tokens > capacity:
        raise ValueError(
            f"MegaMoE got {num_tokens} rows on this rank but the fixed capacity "
            f"is {capacity}; lower the prefill chunk or raise "
            "SGLANG_MEGAMOE_MAX_TOKENS_PER_RANK (A3 limit: 4096)."
        )
    return capacity


def mega_moe_impl() -> str:
    """Provider name passed to the sgl-kernel-npu MegaMoE registry.

    ``auto`` resolves to whatever the installed package registers for this
    generation; the HiSilicon and OTT implementations enforce different
    platform limits, so the provider stays switchable.
    """

    return envs.SGLANG_MEGAMOE_IMPL.get() or "auto"


@functools.lru_cache(maxsize=None)
def _accepts_impl(fn) -> bool:
    """Whether the installed sgl-kernel-npu layer exposes the provider name."""

    try:
        return "impl" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def mega_moe_impl_kwargs(fn) -> dict:
    """Provider selector for packages that ship the impl registry.

    Older sgl-kernel-npu builds bind the operator directly and reject the
    keyword, so it is only forwarded when the installed signature takes it.
    """

    return {"impl": mega_moe_impl()} if _accepts_impl(fn) else {}


def get_mega_moe_sym_buffer(
    runner_config,
    quant_config,
    group,
    num_tokens: int,
    dtype_mode: Optional[str] = None,
):
    """Get a capacity-stable SymmBuffer for one Ascend runner configuration."""

    from sgl_kernel_npu.modules.moe import get_symm_buffer_for_mega_moe

    key = _symm_buffer_key(runner_config, quant_config, group, dtype_mode)
    quant_mode, quant_dtype = key[5], key[6]
    state = _SYMM_BUFFER_CACHE.get(key)
    fixed_capacity = _padded_capacity(num_tokens)
    if fixed_capacity is not None:
        max_tokens = fixed_capacity
    else:
        max_tokens = max(num_tokens, state[0] if state is not None else 1)
    if state is not None and state[0] == max_tokens:
        return state[1]
    if state is None:
        # CANN looks up the EP communicator with init_comm=False. Initialize
        # it here before constructing SymmBuffer, without a device collective.
        device = torch.device("npu", torch.npu.current_device())
        backend = group._get_backend(device)
        ep_rank = torch.distributed.get_rank(group)
        if hasattr(backend, "get_hccl_comm_name"):
            comm_name = backend.get_hccl_comm_name(ep_rank, init_comm=True)
        else:
            comm_name = group.get_hccl_comm_name(ep_rank, init_comm=True)
        if not comm_name:
            raise RuntimeError("MegaMoE EP HCCL comm name is empty after initialization")
    if state is not None and hasattr(state[1], "destroy"):
        state[1].destroy()
    buffer = get_symm_buffer_for_mega_moe(
        group,
        num_experts=key[1],
        num_max_tokens_per_rank=max_tokens,
        num_topk=key[2],
        hidden=key[3],
        intermediate_hidden=key[4],
        dispatch_quant_mode=quant_mode,
        dispatch_quant_out_dtype=quant_dtype,
        **mega_moe_impl_kwargs(get_symm_buffer_for_mega_moe),
    )
    _SYMM_BUFFER_CACHE[key] = (max_tokens, buffer)
    return buffer


def run_mega_moe(
    runner_config,
    quant_config,
    hidden_states,
    topk_weights,
    topk_ids,
    group,
    dtype_mode: Optional[str] = None,
):
    """Run MegaMoE and return the already combined output tensor."""

    if hidden_states.shape[0] == 0:
        raise RuntimeError(
            "ascend_ops fused_op (MegaMoE) does not support an empty batch"
        )

    layer = runner_config.layer
    if layer is None:
        raise RuntimeError("MegaMoE fused-op requires the owning MoE layer")
    check_mega_moe_kernel(layer)

    from sgl_kernel_npu.modules.moe import mega_moe

    # The operator rejects FRACTAL_NZ weights, so prefer the ND copies kept at
    # load time; fall back to the quantizer weights for the INT paths that are
    # still served in place.
    w13_source = getattr(layer, "w13_weight_mega_moe", None)
    w2_source = getattr(layer, "w2_weight_mega_moe", None)
    if w13_source is not None and w2_source is not None:
        # weight1/weight2 are single three-dimensional stacked tensors
        # ([E, out, in/2]); the provider reads the expert count from dim 0.
        w13_weights = [w13_source]
        w2_weights = [w2_source]
        num_local_experts = int(w13_source.shape[0])
    else:
        w13_weights = mega_moe_weight_list(layer.w13_weight)
        w2_weights = mega_moe_weight_list(layer.w2_weight, len(w13_weights))
        num_local_experts = len(w13_weights)
    if len(w2_weights) != len(w13_weights):
        raise ValueError(
            "MegaMoE requires w13_weight and w2_weight to have the same "
            f"expert count, got {len(w13_weights)} and {len(w2_weights)}"
        )

    activation, activation_clamp, activation_params = mega_moe_activation_args(
        runner_config
    )
    mxfp_scales = mega_moe_uses_mxfp_scales(layer)
    # CANN 9.2.0 requires l1_bias/l2_bias to be None on Ascend 950; only the
    # A8W4-INT quantizer carries a real compensation term. 9.1.0 wanted zero
    # tensors for the same arguments, so keep this aligned with the 9.2.0
    # contract from now on.
    l1_bias = mega_moe_bias_list(layer, "w13", num_local_experts)
    l2_bias = mega_moe_bias_list(layer, "w2", num_local_experts)
    mega_moe_kwargs = dict(
        l1_weights_sf=mega_moe_scale_list(
            layer, "w13_weight_scale", num_local_experts, mxfp=mxfp_scales
        ),
        l2_weights_sf=mega_moe_scale_list(
            layer, "w2_weight_scale", num_local_experts, mxfp=mxfp_scales
        ),
        l1_bias=l1_bias,
        l2_bias=l2_bias,
        activation=activation,
        activation_clamp=activation_clamp,
    )
    # MegaMoE expects FP32 routing coefficients, matching the public
    # sgl-kernel-npu contract and the alltoallv combine path.  Keeping the
    # router output in BF16 loses precision before expert outputs are reduced.
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    # The A3 provider does not expose activation_params on the default
    # SwiGLU ABI. Only pass the optional extension when the selected
    # activation actually needs it.
    if activation_params is not None:
        mega_moe_kwargs["activation_params"] = activation_params

    weight_type = mega_moe_weight_type(layer)
    if weight_type is not None:
        mega_moe_kwargs["weight1_type"] = weight_type
        mega_moe_kwargs["weight2_type"] = weight_type

    num_tokens = hidden_states.shape[0]
    pad_capacity = _padded_capacity(num_tokens)
    sym_buffer = get_mega_moe_sym_buffer(
        runner_config, quant_config, group, num_tokens, dtype_mode
    )

    quant_mode, _ = mega_moe_quant_args(quant_config, dtype_mode)
    is_mxfp = quant_mode == 4

    x_active_mask = None
    if pad_capacity is not None and not is_mxfp:
        # A3 expects every rank to reach the fixed capacity with the extra rows
        # marked inactive; MX paths instead pass the batch at its real length.
        if pad_capacity > num_tokens:
            pad_rows = pad_capacity - num_tokens
            hidden_states = torch.cat(
                (
                    hidden_states,
                    hidden_states.new_zeros((pad_rows, hidden_states.shape[1])),
                ),
                dim=0,
            )
            topk_ids = torch.cat(
                (topk_ids, topk_ids.new_zeros((pad_rows, topk_ids.shape[1]))),
                dim=0,
            )
            topk_weights = torch.cat(
                (
                    topk_weights,
                    topk_weights.new_zeros((pad_rows, topk_weights.shape[1])),
                ),
                dim=0,
            )
        x_active_mask = torch.zeros(
            pad_capacity, dtype=torch.int8, device=hidden_states.device
        )
        x_active_mask[:num_tokens] = 1
        mega_moe_kwargs["x_active_mask"] = x_active_mask

    output, _ = mega_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weights,
        w2_weights,
        sym_buffer,
        **mega_moe_impl_kwargs(mega_moe),
        **mega_moe_kwargs,
    )
    return output
