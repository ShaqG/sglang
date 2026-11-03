"""MegaMoE execution adapter for the Ascend MoE runner path."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs


logger = logging.getLogger(__name__)

_SYMM_BUFFER_CACHE: dict[tuple, tuple[int, object]] = {}

# Signatures already logged by ``SGLANG_MEGAMOE_DEBUG``. The operator contract
# only changes when one of the logged fields changes, so one line per shape is
# enough to keep the log readable.
_DEBUG_SEEN: set[tuple] = set()

# Ranks that already emitted the numeric snapshot.
_STATS_SEEN: set = set()

# Ranks that already ran the local-expert probe.
_PROBE_SEEN: set = set()


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


def _marker_pack_scale(scale_int64: torch.Tensor) -> torch.Tensor:
    """Reference ``pack_float_into_int64``: bf16-precision bits plus marker."""

    bits = (scale_int64 & 0xFFFFFFFF) & 0xFFFFE000
    return (bits | (1 << 46)).view(torch.uint64)


def _pack_mega_moe_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert one processed NPU scale tensor to the A3 MegaMoE layout."""

    scale = scale.contiguous()
    marker = envs.SGLANG_MEGAMOE_SCALE_MARKER.get()
    if scale.dtype in (torch.float16, torch.bfloat16, torch.float32):
        scale_fp32 = scale.to(torch.float32).reshape(-1)
        if marker:
            return _marker_pack_scale(scale_fp32.view(torch.int32).to(torch.int64))
        if scale_fp32.numel() % 2:
            raise ValueError(
                "MegaMoE FP scales must contain an even number of values for "
                f"UINT64 packing, got shape {tuple(scale.shape)}"
            )
        return (
            scale_fp32.view(torch.int32).reshape(-1, 2).view(torch.uint64).reshape(-1)
        )
    if scale.dtype in (torch.int64, torch.uint64):
        scale_int64 = scale.view(torch.int64).reshape(-1)
        if marker:
            return _marker_pack_scale(scale_int64)
        return scale_int64.view(torch.uint64)
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


def get_mega_moe_capacity(
    runner_config,
    quant_config,
    group,
    num_tokens: int,
    dtype_mode: Optional[str] = None,
) -> int:
    """Capacity the SymmBuffer uses for a batch of ``num_tokens`` rows."""

    state = _SYMM_BUFFER_CACHE.get(
        _symm_buffer_key(runner_config, quant_config, group, dtype_mode)
    )
    return max(num_tokens, state[0] if state is not None else 1)


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
    )
    _SYMM_BUFFER_CACHE[key] = (max_tokens, buffer)
    return buffer


def _describe_tensors(values) -> str:
    """Shape/dtype summary of a tensor or a per-expert tensor list."""

    if values is None:
        return "None"
    if isinstance(values, (list, tuple)):
        if not values:
            return "empty"
        first = values[0]
    else:
        first = values
    shape = getattr(first, "shape", None)
    dtype = getattr(first, "dtype", None)
    prefix = f"{len(values)}x" if isinstance(values, (list, tuple)) else ""
    return f"{prefix}{tuple(shape) if shape is not None else '?'}/{dtype}"


def _tensor_stats(name: str, value) -> str:
    """Scalar summary used to spot NaN/Inf or wrong magnitudes.

    Only ND tensors are accepted: FRACTAL_NZ weights would need a TransData op,
    which is not safe to insert from this stream context.
    """

    if value is None:
        return f"{name}=None"
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{name}=empty"
        value = value[0]
    try:
        float_value = value.float()
        head = [round(item, 6) for item in float_value.flatten()[:4].tolist()]
        return (
            f"{name}[min={float_value.min().item():.4g}"
            f" max={float_value.max().item():.4g}"
            f" absmean={float_value.abs().mean().item():.4g}"
            f" nan={int(torch.isnan(float_value).sum().item())}"
            f" inf={int(torch.isinf(float_value).sum().item())}"
            f" head={head}]"
        )
    except Exception as exc:  # diagnostics must never break serving
        return f"{name}=<stats failed: {type(exc).__name__}>"


def _log_mega_moe_stats(
    *,
    hidden_states,
    output,
    bias_values,
    bias_values2,
    raw_scale,
    raw_scale2,
) -> None:
    """Log one numeric snapshot per process (diagnostic only)."""

    rank = (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    )
    key = (rank, int(hidden_states.shape[0]))
    if key in _STATS_SEEN or len(_STATS_SEEN) >= 256:
        return
    _STATS_SEEN.add(key)
    parts = [
        _tensor_stats("x", hidden_states),
        _tensor_stats("out", output),
        f"raw_scale={_describe_tensors(raw_scale)}",
        f"raw_scale2={_describe_tensors(raw_scale2)}",
        _tensor_stats("bias", bias_values),
        _tensor_stats("bias2", bias_values2),
    ]
    logger.info("[megamoe-stats] rank=%s %s", rank, " ".join(parts))


def _repack_marker_scales(values):
    if values is None:
        return None
    return [
        _marker_pack_scale(item.view(torch.int64).reshape(-1)) for item in values
    ]


def _swap_last_halves(t: torch.Tensor) -> torch.Tensor:
    half = t.shape[-1] // 2
    return torch.cat((t[..., half:], t[..., :half]), dim=-1).contiguous()


def _swap_w13_layout(weights, scales, bias):
    """Move the gate/up halves of w13 to the other side of the split."""

    from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, npu_format_cast

    swapped_weights = []
    for weight in weights:
        nd = npu_format_cast(weight, NPUACLFormat.ACL_FORMAT_ND)
        nd = _swap_last_halves(nd)
        swapped_weights.append(
            npu_format_cast(nd, NPUACLFormat.ACL_FORMAT_FRACTAL_NZ)
        )
    # cat has no uint64 kernel, so swap through the int64 view.
    swapped_scales = (
        None
        if scales is None
        else [
            _swap_last_halves(item.view(torch.int64)).view(torch.uint64)
            for item in scales
        ]
    )
    swapped_bias = None if bias is None else [_swap_last_halves(item) for item in bias]
    return swapped_weights, swapped_scales, swapped_bias


# Layer id -> swapped w13 tensors. Rebuilt once because the format casts are
# expensive and must not run inside the serving loop.
_SWAPPED_W13_CACHE: dict[int, tuple] = {}


def _apply_mega_moe_variant(name: str, layer_key, w13_weights, mega_moe_kwargs):
    """Rewrite the operator inputs for one alternative interpretation.

    Returns the w13 list to pass; keyword tensors are updated in place. Only
    one variant is applied per run: a mismatched call desynchronises the
    collectives and hangs the scheduler.
    """

    if not name:
        return w13_weights

    if name in ("bias_mul8", "bias_div8", "bias_neg"):
        factor = {"bias_mul8": 8.0, "bias_div8": 1.0 / 8.0, "bias_neg": -1.0}[name]
        for field in ("l1_bias", "l2_bias"):
            values = mega_moe_kwargs.get(field)
            if values is not None:
                mega_moe_kwargs[field] = [item * factor for item in values]
        return w13_weights

    if name == "scale_marker":
        for field in ("l1_weights_sf", "l2_weights_sf"):
            mega_moe_kwargs[field] = _repack_marker_scales(
                mega_moe_kwargs.get(field)
            )
        return w13_weights

    if name == "swap_w13_halves":
        cached = _SWAPPED_W13_CACHE.get(layer_key)
        if cached is None:
            cached = _swap_w13_layout(
                w13_weights,
                mega_moe_kwargs.get("l1_weights_sf"),
                mega_moe_kwargs.get("l1_bias"),
            )
            _SWAPPED_W13_CACHE[layer_key] = cached
        mega_moe_kwargs["l1_weights_sf"] = cached[1]
        mega_moe_kwargs["l1_bias"] = cached[2]
        logger.info("[megamoe-variant] applied %s", name)
        return cached[0]

    if name == "nibble_swap":
        # Element-wise bit swap of the two int4 halves inside every byte, i.e.
        # the opposite nibble order when unpacking the packed weights.
        swapped = []
        for weight in w13_weights:
            int32 = weight.view(torch.int32)
            low = (int32 & 0x0F0F0F0F) << 4
            high = (int32 >> 4) & 0x0F0F0F0F
            swapped.append((low | high).view(weight.dtype))
        logger.info("[megamoe-variant] applied %s", name)
        return swapped

    if name == "reverse_experts":
        for field in ("l1_weights_sf", "l2_weights_sf", "l1_bias", "l2_bias"):
            values = mega_moe_kwargs.get(field)
            if values is not None:
                mega_moe_kwargs[field] = list(reversed(values))
        logger.info("[megamoe-variant] applied %s", name)
        return list(reversed(w13_weights))

    raise ValueError(f"unknown MegaMoE variant {name!r}")


def _log_mega_moe_call(
    *,
    dtype_mode,
    quant_config,
    hidden_states,
    topk_ids,
    topk_weights,
    w13_weights,
    w2_weights,
    scale_values,
    scale_values2,
    bias_values,
    activation,
    activation_clamp,
    activation_params,
    capacity,
    num_tokens,
    mask,
    layer=None,
) -> None:
    """Log the operator contract once per distinct shape (diagnostic only)."""

    quant_mode, quant_dtype = mega_moe_quant_args(quant_config, dtype_mode)
    fields = {
        "mode": dtype_mode,
        "quant": f"({quant_mode}, {quant_dtype})",
        "tokens": num_tokens,
        "rows": int(hidden_states.shape[0]),
        "capacity": capacity,
        "mask": _describe_tensors(mask),
        "x": f"{tuple(hidden_states.shape)}/{hidden_states.dtype}",
        "topk": f"{tuple(topk_ids.shape)}/{topk_ids.dtype}/{topk_weights.dtype}",
        "experts": f"{len(w13_weights)}x{tuple(w13_weights[0].shape)}"
        f"/{w13_weights[0].dtype}",
        "w2": f"{tuple(w2_weights[0].shape)}/{w2_weights[0].dtype}",
        "scales": _describe_tensors(scale_values),
        "scales2": _describe_tensors(scale_values2),
        "bias": _describe_tensors(bias_values),
        "act": activation,
        "clamp": activation_clamp,
        "act_params": activation_params,
        "variant": envs.SGLANG_MEGAMOE_VARIANT.get() or "none",
        "kernel": type(getattr(layer, "w2_kernel", None)).__name__,
        "scale_attrs": (
            ",".join(sorted(name for name in dir(layer) if "scale" in name))
            if layer is not None
            else "?"
        ),
    }
    key = tuple(sorted((name, repr(value)) for name, value in fields.items()))
    if key in _DEBUG_SEEN or len(_DEBUG_SEEN) >= 32:
        return
    _DEBUG_SEEN.add(key)
    rank = (
        torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    )
    logger.info(
        "[megamoe] rank=%s %s",
        rank,
        " ".join(f"{name}={value}" for name, value in fields.items()),
    )


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

    w13_weights = mega_moe_weight_list(layer.w13_weight)
    w2_weights = mega_moe_weight_list(layer.w2_weight, len(w13_weights))
    if len(w2_weights) != len(w13_weights):
        raise ValueError(
            "MegaMoE requires w13_weight and w2_weight to have the same "
            f"expert count, got {len(w13_weights)} and {len(w2_weights)}"
        )

    activation, activation_clamp, activation_params = mega_moe_activation_args(
        runner_config
    )
    mxfp_scales = mega_moe_uses_mxfp_scales(layer)
    l1_bias = mega_moe_bias_list(layer, "w13", len(w13_weights))
    l2_bias = mega_moe_bias_list(layer, "w2", len(w2_weights))
    if mxfp_scales:
        # Packed-FP4 weights have no zero point, but the A5 prototype still
        # lists bias as a required input, so pass zeros.
        l1_bias = l1_bias or [
            torch.zeros(weight.shape[-1], dtype=torch.float32, device=weight.device)
            for weight in w13_weights
        ]
        l2_bias = l2_bias or [
            torch.zeros(weight.shape[-1], dtype=torch.float32, device=weight.device)
            for weight in w2_weights
        ]
    mega_moe_kwargs = dict(
        l1_weights_sf=mega_moe_scale_list(
            layer, "w13_weight_scale", len(w13_weights), mxfp=mxfp_scales
        ),
        l2_weights_sf=mega_moe_scale_list(
            layer, "w2_weight_scale", len(w13_weights), mxfp=mxfp_scales
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

    input_dtype = None
    if envs.SGLANG_MEGAMOE_FP16_ACT.get():
        # The FP4 weight path reports a float16 output prototype, so feed it
        # float16 activations and restore the model dtype afterwards.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float16)

    w13_weights = _apply_mega_moe_variant(
        envs.SGLANG_MEGAMOE_VARIANT.get(), id(layer), w13_weights, mega_moe_kwargs
    )

    num_tokens = hidden_states.shape[0]
    pad_capacity = _padded_capacity(num_tokens)
    sym_buffer = get_mega_moe_sym_buffer(
        runner_config, quant_config, group, num_tokens, dtype_mode
    )

    quant_mode, _ = mega_moe_quant_args(quant_config, dtype_mode)
    is_mxfp = quant_mode == 4

    capacity = pad_capacity
    if capacity is None and envs.SGLANG_MEGAMOE_DEBUG.get():
        capacity = get_mega_moe_capacity(
            runner_config, quant_config, group, num_tokens, dtype_mode
        )

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

    if envs.SGLANG_MEGAMOE_DEBUG.get():
        _log_mega_moe_call(
            dtype_mode=dtype_mode,
            quant_config=quant_config,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w13_weights=w13_weights,
            w2_weights=w2_weights,
            scale_values=mega_moe_kwargs.get("l1_weights_sf"),
            scale_values2=mega_moe_kwargs.get("l2_weights_sf"),
            bias_values=mega_moe_kwargs.get("l1_bias"),
            activation=activation,
            activation_clamp=activation_clamp,
            activation_params=activation_params,
            capacity=capacity,
            num_tokens=num_tokens,
            mask=x_active_mask,
            layer=layer,
        )

    output, _ = mega_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        w13_weights,
        w2_weights,
        sym_buffer,
        **mega_moe_kwargs,
    )
    if input_dtype is not None and output.dtype != input_dtype:
        output = output.to(input_dtype)
    if envs.SGLANG_MEGAMOE_DEBUG.get():
        _log_mega_moe_stats(
            hidden_states=hidden_states,
            output=output,
            bias_values=mega_moe_kwargs.get("l1_bias"),
            bias_values2=mega_moe_kwargs.get("l2_bias"),
            raw_scale=getattr(layer, "w13_weight_scale", None),
            raw_scale2=getattr(layer, "w2_weight_scale", None),
        )

    if envs.SGLANG_MEGAMOE_PROBE.get():
        # Route every row to one local expert with weights summing to one: the
        # result is that expert's full output, independent of routing/combine.
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else None
        )
        if rank not in _PROBE_SEEN:
            _PROBE_SEEN.add(rank)
            ep_rank = torch.distributed.get_rank(group)
            local_experts = len(w13_weights)
            probe_ids = torch.full_like(topk_ids, ep_rank * local_experts)
            probe_weights = torch.full_like(
                topk_weights, 1.0 / max(1, int(topk_ids.shape[1]))
            )
            probe_out, _ = mega_moe(
                hidden_states,
                probe_ids,
                probe_weights,
                w13_weights,
                w2_weights,
                sym_buffer,
                **mega_moe_kwargs,
            )
            logger.info(
                "[megamoe-probe] rank=%s local_expert=%s %s",
                rank,
                ep_rank * local_experts,
                _tensor_stats("out", probe_out),
            )
    return output
