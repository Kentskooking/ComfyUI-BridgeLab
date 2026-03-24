from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def convert_linear_sd15_hypernetwork_to_sdxl(payload: Mapping) -> dict:
    """
    Experimental converter for classic linear SD1.5 hypernetworks.

    The input format is the legacy Automatic1111-style hypernetwork payload that
    ComfyUI loads in nodes_hypernetwork.py. This converter assumes each numeric
    branch is a two-layer linear MLP with no activation, dropout, or layer norm.
    """

    _require_linear_payload(payload)

    output = {key: value for key, value in payload.items() if not _is_dim_key(key)}
    output["name"] = f"{payload.get('name', 'hypernetwork')}_sdxl_projected"
    output["embed_converter_source_hypernetwork"] = str(payload.get("name", ""))
    output["embed_converter_target_arch"] = "sdxl"
    output["embed_converter_projection_strategy"] = "tiled_partial_isometry_affine_lift"

    for branch_index in (0, 1):
        affine_320 = _branch_to_affine(_get_dim_entry(payload, 320)[branch_index], 320)
        affine_640 = _branch_to_affine(_get_dim_entry(payload, 640)[branch_index], 640)
        affine_768 = _branch_to_affine(_get_dim_entry(payload, 768)[branch_index], 768)
        affine_1280 = _branch_to_affine(_get_dim_entry(payload, 1280)[branch_index], 1280)

        lifted_320_to_640 = _project_affine(affine_320, 640)
        combined_640 = _add_affines(affine_640, lifted_320_to_640)
        lifted_640_to_1280 = _project_affine(combined_640, 1280)
        combined_1280 = _add_affines(affine_1280, lifted_640_to_1280)
        lifted_768_to_2048 = _project_affine(affine_768, 2048)

        _set_branch(output, 640, branch_index, _affine_to_branch(combined_640, 640))
        _set_branch(output, 1280, branch_index, _affine_to_branch(combined_1280, 1280))
        _set_branch(output, 2048, branch_index, _affine_to_branch(lifted_768_to_2048, 2048))

    return output


def _require_linear_payload(payload: Mapping) -> None:
    if payload.get("activation_func", "linear") != "linear":
        raise RuntimeError("Only linear hypernetworks are supported by this experimental converter.")
    if payload.get("is_layer_norm", False):
        raise RuntimeError("Layer-norm hypernetworks are not supported by this experimental converter.")
    if payload.get("use_dropout", False):
        raise RuntimeError("Dropout hypernetworks are not supported by this experimental converter.")
    for required_dim in (320, 640, 768, 1280):
        if required_dim not in payload and str(required_dim) not in payload:
            raise RuntimeError(f"Missing required hypernetwork branch: {required_dim}")


def _is_dim_key(key) -> bool:
    try:
        int(key)
        return True
    except (TypeError, ValueError):
        return False


def _set_branch(payload: dict, dim: int, branch_index: int, branch: dict[str, torch.Tensor]) -> None:
    branches = list(payload.get(dim, (None, None)))
    while len(branches) < 2:
        branches.append(None)
    branches[branch_index] = branch
    payload[dim] = tuple(branches)


def _get_dim_entry(payload: Mapping, dim: int):
    if dim in payload:
        return payload[dim]
    dim_key = str(dim)
    if dim_key in payload:
        return payload[dim_key]
    raise RuntimeError(f"Missing required hypernetwork branch: {dim}")


def _branch_to_affine(branch: Mapping[str, torch.Tensor], dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    w1 = branch["linear.0.weight"].to(dtype=torch.float32, device="cpu")
    b1 = branch["linear.0.bias"].to(dtype=torch.float32, device="cpu")
    w2 = branch["linear.1.weight"].to(dtype=torch.float32, device="cpu")
    b2 = branch["linear.1.bias"].to(dtype=torch.float32, device="cpu")

    _validate_branch_shapes(w1, b1, w2, b2, dim)

    linear = w2 @ w1
    bias = (w2 @ b1) + b2
    return linear, bias


def _validate_branch_shapes(
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    dim: int,
) -> None:
    hidden = dim * 2
    expected = {
        "linear.0.weight": (hidden, dim),
        "linear.0.bias": (hidden,),
        "linear.1.weight": (dim, hidden),
        "linear.1.bias": (dim,),
    }
    actual = {
        "linear.0.weight": tuple(w1.shape),
        "linear.0.bias": tuple(b1.shape),
        "linear.1.weight": tuple(w2.shape),
        "linear.1.bias": tuple(b2.shape),
    }
    for key, shape in expected.items():
        if actual[key] != shape:
            raise RuntimeError(f"Unexpected {key} shape for {dim}-branch: {actual[key]} != {shape}")


def _project_affine(affine: tuple[torch.Tensor, torch.Tensor], target_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    source_linear, source_bias = affine
    source_dim = source_linear.shape[0]
    projector = _build_partial_isometry(source_dim, target_dim, dtype=source_linear.dtype)
    reducer = projector.transpose(0, 1)
    target_linear = projector @ source_linear @ reducer
    target_bias = projector @ source_bias
    return target_linear, target_bias


def _build_partial_isometry(source_dim: int, target_dim: int, dtype: torch.dtype) -> torch.Tensor:
    if target_dim < source_dim:
        raise RuntimeError(f"Target dimension {target_dim} must be >= source dimension {source_dim}.")

    counts = [0] * source_dim
    for target_index in range(target_dim):
        counts[target_index % source_dim] += 1

    projector = torch.zeros((target_dim, source_dim), dtype=dtype, device="cpu")
    for target_index in range(target_dim):
        source_index = target_index % source_dim
        projector[target_index, source_index] = 1.0 / math.sqrt(counts[source_index])
    return projector


def _add_affines(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return left[0] + right[0], left[1] + right[1]


def _affine_to_branch(affine: tuple[torch.Tensor, torch.Tensor], dim: int) -> dict[str, torch.Tensor]:
    linear, bias = affine
    hidden = dim * 2
    dtype = linear.dtype

    w1 = torch.zeros((hidden, dim), dtype=dtype, device="cpu")
    w1[:dim, :] = torch.eye(dim, dtype=dtype, device="cpu")
    b1 = torch.zeros((hidden,), dtype=dtype, device="cpu")

    w2 = torch.zeros((dim, hidden), dtype=dtype, device="cpu")
    w2[:, :dim] = linear
    b2 = bias.to(dtype=dtype, device="cpu")

    return {
        "linear.0.weight": w1.contiguous(),
        "linear.0.bias": b1.contiguous(),
        "linear.1.weight": w2.contiguous(),
        "linear.1.bias": b2.contiguous(),
    }
