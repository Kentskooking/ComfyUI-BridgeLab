import hashlib
import os
import re
from dataclasses import dataclass

import safetensors.torch
import torch
from safetensors import safe_open

ALIGNMENT_VERSION = "1"
_CLIP_L_TOKEN_KEYS = (
    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
    "clip_l.transformer.text_model.embeddings.token_embedding.weight",
)
_CLIP_G_TOKEN_KEYS = (
    "conditioner.embedders.1.model.token_embedding.weight",
    "clip_g.transformer.text_model.embeddings.token_embedding.weight",
)


@dataclass
class AlignmentTransform:
    matrix: torch.Tensor
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    scale: float
    fit_token_cosine: float
    checkpoint_name: str
    checkpoint_fingerprint: str

    def map_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        source_mean = self.source_mean.to(device=vectors.device, dtype=torch.float32)
        target_mean = self.target_mean.to(device=vectors.device, dtype=torch.float32)
        matrix = self.matrix.to(device=vectors.device, dtype=torch.float32)
        vectors = vectors.to(dtype=torch.float32)
        return (vectors - source_mean) @ matrix + target_mean


def compute_checkpoint_fingerprint(checkpoint_path: str) -> str:
    stat = os.stat(checkpoint_path)
    payload = "|".join(
        [
            ALIGNMENT_VERSION,
            os.path.abspath(checkpoint_path),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "alignment"


def alignment_cache_path(cache_root: str, checkpoint_name: str, checkpoint_fingerprint: str) -> str:
    filename = f"{safe_stem(checkpoint_name)}_{checkpoint_fingerprint}.safetensors"
    return os.path.join(cache_root, filename)


def load_alignment_cache(cache_path: str) -> AlignmentTransform | None:
    if not os.path.isfile(cache_path):
        return None

    try:
        tensors = safetensors.torch.load_file(cache_path, device="cpu")
        with safe_open(cache_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception:
        return None

    required_metadata = (
        "embed_converter.version",
        "embed_converter.scale",
        "embed_converter.fit_token_cosine",
        "embed_converter.checkpoint_name",
        "embed_converter.checkpoint_fingerprint",
    )
    required_tensors = ("matrix", "source_mean", "target_mean")
    if metadata.get("embed_converter.version") != ALIGNMENT_VERSION:
        return None
    if any(key not in metadata for key in required_metadata):
        return None
    if any(key not in tensors for key in required_tensors):
        return None

    return AlignmentTransform(
        matrix=tensors["matrix"],
        source_mean=tensors["source_mean"],
        target_mean=tensors["target_mean"],
        scale=float(metadata["embed_converter.scale"]),
        fit_token_cosine=float(metadata["embed_converter.fit_token_cosine"]),
        checkpoint_name=metadata["embed_converter.checkpoint_name"],
        checkpoint_fingerprint=metadata["embed_converter.checkpoint_fingerprint"],
    )


def save_alignment_cache(cache_path: str, alignment: AlignmentTransform) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    metadata = {
        "embed_converter.version": ALIGNMENT_VERSION,
        "embed_converter.scale": f"{alignment.scale:.12f}",
        "embed_converter.fit_token_cosine": f"{alignment.fit_token_cosine:.12f}",
        "embed_converter.checkpoint_name": alignment.checkpoint_name,
        "embed_converter.checkpoint_fingerprint": alignment.checkpoint_fingerprint,
    }
    tensors = {
        "matrix": alignment.matrix.cpu(),
        "source_mean": alignment.source_mean.cpu(),
        "target_mean": alignment.target_mean.cpu(),
    }
    safetensors.torch.save_file(tensors, cache_path, metadata=metadata)


def fit_scaled_orthogonal_alignment(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    checkpoint_name: str,
    checkpoint_fingerprint: str,
) -> AlignmentTransform:
    source = source_embeddings.to(dtype=torch.float32, device="cpu")
    target = target_embeddings.to(dtype=torch.float32, device="cpu")
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    # Rectangular Procrustes via the thin SVD of the cross-covariance.
    cross_covariance = source_centered.T @ target_centered
    u, singular_values, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    rotation = u @ vh
    scale = float(singular_values.sum() / source_centered.square().sum().clamp_min(1e-12))
    matrix = rotation * scale
    fit_token_cosine = mean_cosine_similarity(
        source,
        target,
        lambda batch: (batch - source_mean) @ matrix + target_mean,
    )

    return AlignmentTransform(
        matrix=matrix.cpu(),
        source_mean=source_mean.cpu(),
        target_mean=target_mean.cpu(),
        scale=scale,
        fit_token_cosine=fit_token_cosine,
        checkpoint_name=checkpoint_name,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )


def mean_cosine_similarity(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    mapper,
    batch_size: int = 4096,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, source_embeddings.shape[0], batch_size):
        stop = min(start + batch_size, source_embeddings.shape[0])
        mapped = mapper(source_embeddings[start:stop])
        target = target_embeddings[start:stop]
        cosine = torch.nn.functional.cosine_similarity(mapped, target, dim=-1)
        total += cosine.sum().item()
        count += cosine.numel()
    if count == 0:
        return 0.0
    return total / count


def load_checkpoint_token_tables(checkpoint_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    if checkpoint_path.lower().endswith((".safetensors", ".sft")):
        return _load_token_tables_from_safetensors(checkpoint_path)
    return _load_token_tables_from_torch_file(checkpoint_path)


def _load_token_tables_from_safetensors(checkpoint_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        clip_l_key = next((key for key in _CLIP_L_TOKEN_KEYS if key in keys), None)
        clip_g_key = next((key for key in _CLIP_G_TOKEN_KEYS if key in keys), None)
        if clip_l_key is None or clip_g_key is None:
            raise RuntimeError(
                "The selected checkpoint does not expose SDXL CLIP-L and CLIP-G token embedding tables."
            )
        clip_l = handle.get_tensor(clip_l_key).clone()
        clip_g = handle.get_tensor(clip_g_key).clone()
    return clip_l, clip_g


def _load_token_tables_from_torch_file(checkpoint_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    import comfy.utils

    state_dict = comfy.utils.load_torch_file(checkpoint_path, safe_load=True, device=torch.device("cpu"))
    clip_l = _find_state_dict_tensor(state_dict, _CLIP_L_TOKEN_KEYS)
    clip_g = _find_state_dict_tensor(state_dict, _CLIP_G_TOKEN_KEYS)
    if clip_l is None or clip_g is None:
        raise RuntimeError(
            "The selected checkpoint does not expose SDXL CLIP-L and CLIP-G token embedding tables."
        )
    return clip_l.clone(), clip_g.clone()


def _find_state_dict_tensor(state_dict: dict, candidate_keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None:
            return tensor
    return None
