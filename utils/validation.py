import gc
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sd
import folder_paths


@dataclass
class PhraseValidationResult:
    hidden_cosine: float
    pooled_cosine: float
    phrase_count: int
    phrase_source: str
    ran: bool

    @classmethod
    def skipped(cls) -> "PhraseValidationResult":
        return cls(hidden_cosine=-1.0, pooled_cosine=-1.0, phrase_count=0, phrase_source="skipped", ran=False)


def validate_alignment_with_ci_phrases(
    checkpoint_path: str,
    alignment,
    clip_l_token_table: torch.Tensor,
    phrase_count: int,
    batch_size: int,
) -> PhraseValidationResult:
    phrases = sample_phrases(load_clip_interrogator_phrases(), phrase_count)
    if not phrases:
        raise RuntimeError("No CLIP Interrogator phrases were available for validation.")

    _, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
        checkpoint_path,
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        output_model=False,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    if clip is None:
        raise RuntimeError("Failed to load the checkpoint CLIP encoders for phrase validation.")

    hidden_total = 0.0
    hidden_weight = 0.0
    pooled_total = 0.0
    phrase_total = 0

    try:
        for start in range(0, len(phrases), batch_size):
            batch_phrases = phrases[start : start + batch_size]
            for phrase in batch_phrases:
                tokens = clip.tokenize(phrase)
                tokens_l = tokens["l"]
                tokens_g = tokens["g"]

                clip.load_model({"l": tokens_l, "g": tokens_g})
                clip.cond_stage_model.clip_g.reset_clip_options()
                clip.cond_stage_model.clip_g.set_clip_options({"execution_device": clip.patcher.load_device})

                mapped_tokens = build_mapped_clip_g_tokens(tokens_l, tokens_g, clip_l_token_table, alignment)
                with torch.no_grad():
                    true_hidden, true_pooled = clip.cond_stage_model.clip_g.encode_token_weights(tokens_g)
                    mapped_hidden, mapped_pooled = clip.cond_stage_model.clip_g.encode_token_weights(mapped_tokens)

                mask = flattened_content_mask(tokens_g, device=true_hidden.device)
                if mask.sum().item() == 0:
                    mask = flattened_non_pad_mask(tokens_g, device=true_hidden.device)

                hidden_cosine = F.cosine_similarity(true_hidden, mapped_hidden, dim=-1)
                hidden_total += (hidden_cosine * mask).sum().item()
                hidden_weight += mask.sum().item()
                pooled_total += F.cosine_similarity(true_pooled, mapped_pooled, dim=-1).sum().item()
                phrase_total += 1
    finally:
        del clip
        gc.collect()
        comfy.model_management.soft_empty_cache()

    hidden_score = hidden_total / hidden_weight if hidden_weight else 0.0
    pooled_score = pooled_total / phrase_total if phrase_total else 0.0
    return PhraseValidationResult(
        hidden_cosine=hidden_score,
        pooled_cosine=pooled_score,
        phrase_count=phrase_total,
        phrase_source="clip_interrogator",
        ran=True,
    )


def build_mapped_clip_g_tokens(
    batch_tokens_l,
    batch_tokens_g,
    clip_l_token_table: torch.Tensor,
    alignment,
):
    mapped_batch = []
    content_token_ids = []
    content_locations = []

    for sequence_index, (sequence_l, sequence_g) in enumerate(zip(batch_tokens_l, batch_tokens_g)):
        mapped_sequence = []
        for token_index, ((token_l, _weight_l), (token_g, weight_g)) in enumerate(zip(sequence_l, sequence_g)):
            if keep_native_clip_g_token(token_g):
                mapped_sequence.append((token_g, weight_g))
                continue
            if not isinstance(token_l, int):
                raise RuntimeError("Phrase validation only supports integer CLIP-L tokens.")
            content_locations.append((sequence_index, token_index, weight_g))
            content_token_ids.append(token_l)
            mapped_sequence.append(None)
        mapped_batch.append(mapped_sequence)

    if content_token_ids:
        token_ids = torch.tensor(content_token_ids, dtype=torch.long)
        mapped_vectors = alignment.map_vectors(clip_l_token_table.index_select(0, token_ids))
        for mapped_vector, (sequence_index, token_index, weight) in zip(mapped_vectors, content_locations):
            mapped_batch[sequence_index][token_index] = (mapped_vector, weight)

    return mapped_batch


def keep_native_clip_g_token(token_id) -> bool:
    return not isinstance(token_id, int) or token_id in (0, 49406, 49407)


def content_mask(batch_tokens_g, device: torch.device) -> torch.Tensor:
    rows = []
    for sequence in batch_tokens_g:
        rows.append(
            [0.0 if keep_native_clip_g_token(token) else 1.0 for token, _weight in sequence]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def non_pad_mask(batch_tokens_g, device: torch.device) -> torch.Tensor:
    rows = []
    for sequence in batch_tokens_g:
        rows.append([0.0 if (isinstance(token, int) and token == 0) else 1.0 for token, _weight in sequence])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def flattened_content_mask(batch_tokens_g, device: torch.device) -> torch.Tensor:
    return content_mask(batch_tokens_g, device=device).reshape(1, -1)


def flattened_non_pad_mask(batch_tokens_g, device: torch.device) -> torch.Tensor:
    return non_pad_mask(batch_tokens_g, device=device).reshape(1, -1)


def sample_phrases(phrases: list[str], phrase_count: int) -> list[str]:
    if phrase_count <= 0 or phrase_count >= len(phrases):
        return list(phrases)
    step = len(phrases) / float(phrase_count)
    sampled = []
    seen = set()
    for index in range(phrase_count):
        phrase = phrases[min(int(index * step), len(phrases) - 1)]
        if phrase not in seen:
            seen.add(phrase)
            sampled.append(phrase)
    return sampled


def load_clip_interrogator_phrases() -> list[str]:
    data_path = clip_interrogator_data_path()
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"CLIP Interrogator data path not found: {data_path}")

    artists_raw = load_phrase_file(os.path.join(data_path, "artists.txt"))
    flavors = load_phrase_file(os.path.join(data_path, "flavors.txt"))
    mediums = load_phrase_file(os.path.join(data_path, "mediums.txt"))
    movements = load_phrase_file(os.path.join(data_path, "movements.txt"))

    artists = [f"by {artist}" for artist in artists_raw]
    artists.extend([f"inspired by {artist}" for artist in artists_raw])

    sites = [
        "Artstation",
        "behance",
        "cg society",
        "cgsociety",
        "deviantart",
        "dribbble",
        "flickr",
        "instagram",
        "pexels",
        "pinterest",
        "pixabay",
        "pixiv",
        "polycount",
        "reddit",
        "shutterstock",
        "tumblr",
        "unsplash",
        "zbrush central",
    ]
    trendings = [site for site in sites]
    trendings.extend([f"trending on {site}" for site in sites])
    trendings.extend([f"featured on {site}" for site in sites])
    trendings.extend([f"{site} contest winner" for site in sites])

    return artists + flavors + mediums + movements + trendings


def clip_interrogator_data_path() -> str:
    custom_nodes_root = folder_paths.get_folder_paths("custom_nodes")[0]
    return os.path.join(custom_nodes_root, "eden_comfy_pipelines", "clip_utils", "data")


def load_phrase_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip() for line in handle if line.strip()]
