import os
import re
from dataclasses import dataclass

import safetensors.torch
import torch
from safetensors import safe_open


@dataclass
class EmbeddingBundle:
    clip_l: torch.Tensor | None
    clip_g: torch.Tensor | None
    metadata: dict[str, str]
    source_path: str

    @property
    def token_count(self) -> int:
        if self.clip_l is not None:
            return int(self.clip_l.shape[0])
        if self.clip_g is not None:
            return int(self.clip_g.shape[0])
        return 0

    @property
    def is_sdxl(self) -> bool:
        return self.clip_l is not None and self.clip_g is not None


def load_embedding_bundle(embedding_path: str) -> EmbeddingBundle:
    if embedding_path.lower().endswith((".safetensors", ".sft")):
        return _load_safetensors_embedding(embedding_path)
    return _load_pt_embedding(embedding_path)


def normalize_output_name(output_name: str, fallback_stem: str) -> str:
    name = (output_name or "").strip()
    if not name:
        name = fallback_stem
    stem = os.path.splitext(os.path.basename(name))[0]
    return normalize_output_component(stem, fallback_stem)


def normalize_output_component(name: str, fallback_component: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._")
    return component or fallback_component


def build_output_path(output_dir: str, output_name: str, fallback_stem: str, overwrite: bool) -> tuple[str, str]:
    embedding_name = normalize_output_name(output_name, fallback_stem)
    output_path = os.path.join(output_dir, f"{embedding_name}.safetensors")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output embedding already exists: {output_path}")
    return output_path, embedding_name


def save_sdxl_embedding(
    output_path: str,
    clip_l: torch.Tensor,
    clip_g: torch.Tensor,
    metadata: dict[str, str],
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safetensors.torch.save_file(
        {"clip_l": clip_l.contiguous().cpu(), "clip_g": clip_g.contiguous().cpu()},
        output_path,
        metadata=metadata,
    )


def _load_safetensors_embedding(embedding_path: str) -> EmbeddingBundle:
    tensors = {}
    with safe_open(embedding_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).clone()

    clip_l = _coerce_tensor(tensors.get("clip_l"))
    clip_g = _coerce_tensor(tensors.get("clip_g"))
    if clip_l is None and clip_g is None:
        inferred_key, inferred_tensor = _infer_single_tensor(tensors)
        if inferred_key == "clip_l":
            clip_l = inferred_tensor
        elif inferred_key == "clip_g":
            clip_g = inferred_tensor

    return EmbeddingBundle(clip_l=clip_l, clip_g=clip_g, metadata=dict(metadata), source_path=embedding_path)


def _load_pt_embedding(embedding_path: str) -> EmbeddingBundle:
    import comfy.utils

    payload = comfy.utils.load_torch_file(embedding_path, safe_load=True, device=torch.device("cpu"))
    metadata = {}
    clip_l = None
    clip_g = None

    if isinstance(payload, dict):
        for key in ("name", "step", "sd_checkpoint", "sd_checkpoint_name"):
            value = payload.get(key)
            if value is not None:
                metadata[f"legacy.{key}"] = str(value)

        if "clip_l" in payload:
            clip_l = _coerce_tensor(payload["clip_l"])
        if "clip_g" in payload:
            clip_g = _coerce_tensor(payload["clip_g"])

        string_to_param = payload.get("string_to_param")
        if isinstance(string_to_param, dict) and (clip_l is None and clip_g is None):
            first_tensor = next(iter(string_to_param.values()), None)
            inferred = _coerce_tensor(first_tensor)
            if inferred is not None:
                if inferred.shape[-1] == 768:
                    clip_l = inferred
                elif inferred.shape[-1] == 1280:
                    clip_g = inferred

    if clip_l is None and clip_g is None:
        raise RuntimeError(f"Unsupported embedding format: {embedding_path}")

    return EmbeddingBundle(clip_l=clip_l, clip_g=clip_g, metadata=metadata, source_path=embedding_path)


def _infer_single_tensor(tensors: dict[str, torch.Tensor]) -> tuple[str | None, torch.Tensor | None]:
    for tensor in tensors.values():
        inferred = _coerce_tensor(tensor)
        if inferred is None:
            continue
        if inferred.shape[-1] == 768:
            return "clip_l", inferred
        if inferred.shape[-1] == 1280:
            return "clip_g", inferred
    return None, None


def _coerce_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or not torch.is_tensor(tensor):
        return None
    if tensor.ndim == 1:
        return tensor.view(1, -1)
    if tensor.ndim >= 2:
        return tensor.reshape(-1, tensor.shape[-1])
    return None
