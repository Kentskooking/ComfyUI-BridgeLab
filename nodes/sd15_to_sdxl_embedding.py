import gc
import os
from datetime import datetime, timezone

import torch

import comfy.model_management
import comfy.utils
import folder_paths

from ..utils.alignment import (
    alignment_cache_path,
    compute_checkpoint_fingerprint,
    fit_scaled_orthogonal_alignment,
    load_alignment_cache,
    load_checkpoint_token_tables,
    save_alignment_cache,
)
from ..utils.embedding_io import (
    build_output_path,
    load_embedding_bundle,
    normalize_output_component,
    normalize_output_name,
    save_sdxl_embedding,
)
from ..utils.hypernetwork_projection import convert_linear_sd15_hypernetwork_to_sdxl
from ..utils.validation import PhraseValidationResult, validate_alignment_with_ci_phrases

CONVERTED_BATCH_DIRNAME = "converted_sd15-SDXL"
MAX_REPORT_LINES = 100


class SD15ToSDXLEmbeddingConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd15_embedding": (folder_paths.get_filename_list("embeddings"),),
                "sdxl_checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "output_name": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "run_ci_validation": ("BOOLEAN", {"default": True}),
                "validation_phrase_count": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 16}),
                "validation_batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "embedding_name",
        "embedding_path",
        "fit_token_cosine",
        "validation_hidden_cosine",
        "validation_pooled_cosine",
        "report",
    )
    FUNCTION = "convert_embedding"
    OUTPUT_NODE = True
    CATEGORY = "embed_converter"
    DESCRIPTION = (
        "Converts an SD 1.5 textual inversion embedding into an SDXL embedding by keeping the "
        "CLIP-L vectors and projecting them into CLIP-G with a cached token-table Procrustes fit."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node writes files, so it should not be skipped by execution cache reuse.
        return float("nan")

    def convert_embedding(
        self,
        sd15_embedding,
        sdxl_checkpoint,
        output_name,
        overwrite,
        run_ci_validation,
        validation_phrase_count,
        validation_batch_size,
    ):
        embedding_path = folder_paths.get_full_path_or_raise("embeddings", sd15_embedding)
        checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", sdxl_checkpoint)
        embedding_bundle = load_embedding_bundle(embedding_path)
        validate_sd15_embedding_bundle(embedding_bundle, sd15_embedding)

        output_dir = folder_paths.get_folder_paths("embeddings")[0]
        fallback_stem = f"{os.path.splitext(os.path.basename(sd15_embedding))[0]}_sdxl"
        output_path, embedding_name = build_output_path(output_dir, output_name, fallback_stem, overwrite)

        alignment, clip_l_token_table, clip_g_token_table, cache_path = prepare_alignment(
            checkpoint_path,
            sdxl_checkpoint,
        )
        validation = run_alignment_validation(
            checkpoint_path,
            alignment,
            clip_l_token_table,
            run_ci_validation,
            validation_phrase_count,
            validation_batch_size,
        )
        converted_clip_l, converted_clip_g = convert_embedding_bundle(
            embedding_bundle,
            clip_l_token_table,
            clip_g_token_table,
            alignment,
        )

        metadata = build_embedding_metadata(
            embedding_name=embedding_name,
            source_embedding=sd15_embedding,
            checkpoint_name=sdxl_checkpoint,
            alignment=alignment,
            token_count=embedding_bundle.token_count,
            validation=validation,
            source_metadata=embedding_bundle.metadata,
        )
        save_sdxl_embedding(output_path, converted_clip_l, converted_clip_g, metadata)
        folder_paths.filename_list_cache.pop("embeddings", None)

        report = build_report(
            embedding_name=embedding_name,
            output_path=output_path,
            alignment=alignment,
            token_count=embedding_bundle.token_count,
            validation=validation,
            cache_path=cache_path,
        )

        del clip_l_token_table
        del clip_g_token_table
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return (
            embedding_name,
            output_path,
            float(alignment.fit_token_cosine),
            float(validation.hidden_cosine),
            float(validation.pooled_cosine),
            report,
        )


class SD15FolderToSDXLEmbeddingBatchConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder_path": ("STRING", {"default": folder_paths.get_folder_paths("embeddings")[0]}),
                "sdxl_checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "name_suffix": ("STRING", {"default": "_sdxl"}),
                "recursive": ("BOOLEAN", {"default": False}),
                "overwrite": ("BOOLEAN", {"default": False}),
                "run_ci_validation": ("BOOLEAN", {"default": True}),
                "validation_phrase_count": ("INT", {"default": 128, "min": 16, "max": 4096, "step": 16}),
                "validation_batch_size": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "output_directory",
        "converted_count",
        "skipped_count",
        "failed_count",
        "fit_token_cosine",
        "validation_hidden_cosine",
        "validation_pooled_cosine",
        "report",
    )
    FUNCTION = "convert_folder"
    OUTPUT_NODE = True
    CATEGORY = "embed_converter"
    DESCRIPTION = (
        "Converts every SD 1.5 embedding found in a folder into SDXL safetensors and writes them into "
        "models/embeddings/converted_sd15-SDXL, preserving source subfolders."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def convert_folder(
        self,
        input_folder_path,
        sdxl_checkpoint,
        name_suffix,
        recursive,
        overwrite,
        run_ci_validation,
        validation_phrase_count,
        validation_batch_size,
    ):
        input_folder = os.path.abspath(os.path.expanduser((input_folder_path or "").strip()))
        if not os.path.isdir(input_folder):
            raise RuntimeError(f"Input folder does not exist: {input_folder}")

        output_directory = batch_output_directory()
        checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", sdxl_checkpoint)
        alignment, clip_l_token_table, clip_g_token_table, cache_path = prepare_alignment(
            checkpoint_path,
            sdxl_checkpoint,
        )
        validation = run_alignment_validation(
            checkpoint_path,
            alignment,
            clip_l_token_table,
            run_ci_validation,
            validation_phrase_count,
            validation_batch_size,
        )

        converted = []
        skipped = []
        failed = []
        for source_path in iter_embedding_files(input_folder, recursive, exclude_root=output_directory):
            relative_source = os.path.relpath(source_path, input_folder)
            try:
                bundle = load_embedding_bundle(source_path)
                validate_sd15_embedding_bundle(bundle, relative_source)
                output_path, embedding_name, relative_output = build_batch_output_path(
                    output_directory,
                    relative_source,
                    name_suffix,
                    overwrite,
                )
                converted_clip_l, converted_clip_g = convert_embedding_bundle(
                    bundle,
                    clip_l_token_table,
                    clip_g_token_table,
                    alignment,
                )
                metadata = build_embedding_metadata(
                    embedding_name=embedding_name,
                    source_embedding=relative_source,
                    checkpoint_name=sdxl_checkpoint,
                    alignment=alignment,
                    token_count=bundle.token_count,
                    validation=validation,
                    source_metadata=bundle.metadata,
                )
                save_sdxl_embedding(output_path, converted_clip_l, converted_clip_g, metadata)
                converted.append(f"{relative_source} -> {relative_output}")
            except FileExistsError:
                skipped.append(f"{relative_source} -> exists")
            except RuntimeError as exc:
                skipped.append(f"{relative_source} -> {exc}")
            except Exception as exc:
                failed.append(f"{relative_source} -> {type(exc).__name__}: {exc}")

        folder_paths.filename_list_cache.pop("embeddings", None)
        report = build_batch_report(
            output_directory=output_directory,
            alignment=alignment,
            validation=validation,
            cache_path=cache_path,
            converted=converted,
            skipped=skipped,
            failed=failed,
        )

        del clip_l_token_table
        del clip_g_token_table
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return (
            output_directory,
            len(converted),
            len(skipped),
            len(failed),
            float(alignment.fit_token_cosine),
            float(validation.hidden_cosine),
            float(validation.pooled_cosine),
            report,
        )


class SD15ToSDXLHypernetworkConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sd15_hypernetwork": (folder_paths.get_filename_list("hypernetworks"),),
                "output_name": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("hypernetwork_name", "hypernetwork_path", "report")
    FUNCTION = "convert_hypernetwork"
    OUTPUT_NODE = True
    CATEGORY = "embed_converter"
    DESCRIPTION = (
        "Experimental converter that lifts a classic linear SD 1.5 hypernetwork into SDXL-sized "
        "640/1280/2048 branches and saves it as a .pt hypernetwork."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def convert_hypernetwork(self, sd15_hypernetwork, output_name, overwrite):
        hypernetwork_path = folder_paths.get_full_path_or_raise("hypernetworks", sd15_hypernetwork)
        payload = comfy.utils.load_torch_file(hypernetwork_path, safe_load=True, device=torch.device("cpu"))
        converted = convert_linear_sd15_hypernetwork_to_sdxl(payload)

        output_dir = folder_paths.get_folder_paths("hypernetworks")[0]
        fallback_stem = f"{os.path.splitext(os.path.basename(sd15_hypernetwork))[0]}_SDXL_projected"
        output_path, hypernetwork_name = build_hypernetwork_output_path(output_dir, output_name, fallback_stem, overwrite)
        converted["name"] = hypernetwork_name

        save_hypernetwork_payload(output_path, converted)
        folder_paths.filename_list_cache.pop("hypernetworks", None)

        report = build_hypernetwork_report(
            hypernetwork_name=hypernetwork_name,
            output_path=output_path,
            source_hypernetwork=sd15_hypernetwork,
            payload=converted,
        )

        gc.collect()
        comfy.model_management.soft_empty_cache()

        return (
            hypernetwork_name,
            output_path,
            report,
        )


def build_embedding_metadata(
    embedding_name,
    source_embedding,
    checkpoint_name,
    alignment,
    token_count,
    validation,
    source_metadata,
):
    metadata = {
        "modelspec.architecture": "stable-diffusion-xl-v1-base/textual-inversion",
        "modelspec.implementation": "https://github.com/comfyanonymous/ComfyUI",
        "modelspec.title": embedding_name,
        "modelspec.date": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "embed_converter.source_embedding": source_embedding,
        "embed_converter.target_checkpoint": checkpoint_name,
        "embed_converter.alignment_version": "1",
        "embed_converter.checkpoint_fingerprint": alignment.checkpoint_fingerprint,
        "embed_converter.token_count": str(token_count),
        "embed_converter.fit_token_cosine": f"{alignment.fit_token_cosine:.12f}",
    }
    if validation.ran:
        metadata["embed_converter.validation_phrase_source"] = validation.phrase_source
        metadata["embed_converter.validation_phrase_count"] = str(validation.phrase_count)
        metadata["embed_converter.validation_hidden_cosine"] = f"{validation.hidden_cosine:.12f}"
        metadata["embed_converter.validation_pooled_cosine"] = f"{validation.pooled_cosine:.12f}"
    for key, value in (source_metadata or {}).items():
        metadata[f"embed_converter.source_meta.{key}"] = str(value)
    return metadata


def build_report(embedding_name, output_path, alignment, token_count, validation, cache_path):
    lines = [
        f"saved_embedding={embedding_name}",
        f"output_path={output_path}",
        f"token_count={token_count}",
        f"fit_token_cosine={alignment.fit_token_cosine:.6f}",
        f"alignment_cache={cache_path}",
    ]
    if validation.ran:
        lines.append(f"validation_phrase_count={validation.phrase_count}")
        lines.append(f"validation_hidden_cosine={validation.hidden_cosine:.6f}")
        lines.append(f"validation_pooled_cosine={validation.pooled_cosine:.6f}")
    else:
        lines.append("validation=skipped")
    report = "\n".join(lines)
    print(report)
    return report


def prepare_alignment(checkpoint_path, checkpoint_name):
    checkpoint_fingerprint = compute_checkpoint_fingerprint(checkpoint_path)
    cache_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "alignments")
    cache_path = alignment_cache_path(cache_root, checkpoint_name, checkpoint_fingerprint)

    clip_l_token_table, clip_g_token_table = load_checkpoint_token_tables(checkpoint_path)
    alignment = load_alignment_cache(cache_path)
    if alignment is None:
        alignment = fit_scaled_orthogonal_alignment(
            clip_l_token_table,
            clip_g_token_table,
            checkpoint_name=checkpoint_name,
            checkpoint_fingerprint=checkpoint_fingerprint,
        )
        save_alignment_cache(cache_path, alignment)
    return alignment, clip_l_token_table, clip_g_token_table, cache_path


def run_alignment_validation(
    checkpoint_path,
    alignment,
    clip_l_token_table,
    run_ci_validation,
    validation_phrase_count,
    validation_batch_size,
):
    if not run_ci_validation:
        return PhraseValidationResult.skipped()
    return validate_alignment_with_ci_phrases(
        checkpoint_path,
        alignment,
        clip_l_token_table.to(dtype=torch.float32, device="cpu"),
        phrase_count=validation_phrase_count,
        batch_size=validation_batch_size,
    )


def convert_embedding_bundle(embedding_bundle, clip_l_token_table, clip_g_token_table, alignment):
    converted_clip_l = embedding_bundle.clip_l.to(dtype=clip_l_token_table.dtype)
    converted_clip_g = alignment.map_vectors(embedding_bundle.clip_l).to(dtype=clip_g_token_table.dtype)
    return converted_clip_l, converted_clip_g


def validate_sd15_embedding_bundle(embedding_bundle, source_name):
    if embedding_bundle.clip_l is None:
        raise RuntimeError(f"{source_name} does not contain CLIP-L vectors")
    if embedding_bundle.clip_l.shape[-1] != 768:
        raise RuntimeError(
            f"{source_name} has CLIP-L width {embedding_bundle.clip_l.shape[-1]} instead of 768"
        )
    if embedding_bundle.clip_g is not None:
        raise RuntimeError(f"{source_name} already contains CLIP-G vectors")


def batch_output_directory():
    base_embeddings_dir = folder_paths.get_folder_paths("embeddings")[0]
    output_dir = os.path.join(base_embeddings_dir, CONVERTED_BATCH_DIRNAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def iter_embedding_files(input_folder, recursive, exclude_root=None):
    allowed_extensions = {ext.lower() for ext in folder_paths.supported_pt_extensions}
    if recursive:
        for root, dirs, files in os.walk(input_folder):
            if exclude_root and same_or_nested_path(root, exclude_root):
                dirs[:] = []
                continue
            for filename in sorted(files):
                extension = os.path.splitext(filename)[1].lower()
                if extension in allowed_extensions:
                    yield os.path.join(root, filename)
        return

    for filename in sorted(os.listdir(input_folder)):
        path = os.path.join(input_folder, filename)
        if not os.path.isfile(path):
            continue
        if exclude_root and same_or_nested_path(path, exclude_root):
            continue
        extension = os.path.splitext(filename)[1].lower()
        if extension in allowed_extensions:
            yield path


def same_or_nested_path(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def build_batch_output_path(output_root, relative_source_path, name_suffix, overwrite):
    relative_source_path = os.path.normpath(relative_source_path)
    relative_dir = os.path.dirname(relative_source_path)
    source_stem = os.path.splitext(os.path.basename(relative_source_path))[0]
    embedding_name = normalize_output_name(f"{source_stem}{name_suffix}", "converted_embedding")

    output_dir = output_root
    relative_output_parts = []
    if relative_dir and relative_dir != ".":
        for index, part in enumerate(relative_dir.split(os.sep), start=1):
            if not part or part == ".":
                continue
            normalized_part = normalize_output_component(part, f"folder_{index}")
            relative_output_parts.append(normalized_part)
        if relative_output_parts:
            output_dir = os.path.join(output_root, *relative_output_parts)

    filename = f"{embedding_name}.safetensors"
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output embedding already exists: {output_path}")

    relative_output = os.path.join(*relative_output_parts, filename) if relative_output_parts else filename
    return output_path, embedding_name, relative_output


def build_batch_report(output_directory, alignment, validation, cache_path, converted, skipped, failed):
    lines = [
        f"output_directory={output_directory}",
        f"converted_count={len(converted)}",
        f"skipped_count={len(skipped)}",
        f"failed_count={len(failed)}",
        f"fit_token_cosine={alignment.fit_token_cosine:.6f}",
        f"alignment_cache={cache_path}",
    ]
    if validation.ran:
        lines.append(f"validation_phrase_count={validation.phrase_count}")
        lines.append(f"validation_hidden_cosine={validation.hidden_cosine:.6f}")
        lines.append(f"validation_pooled_cosine={validation.pooled_cosine:.6f}")
    else:
        lines.append("validation=skipped")

    lines.extend(report_section("converted", converted))
    lines.extend(report_section("skipped", skipped))
    lines.extend(report_section("failed", failed))

    report = "\n".join(lines)
    print(report)
    return report


def report_section(section_name, items):
    if not items:
        return [f"{section_name}=none"]
    lines = [f"{section_name}_items={len(items)}"]
    for item in items[:MAX_REPORT_LINES]:
        lines.append(f"{section_name}:{item}")
    if len(items) > MAX_REPORT_LINES:
        lines.append(f"{section_name}:... truncated {len(items) - MAX_REPORT_LINES} more")
    return lines


def build_hypernetwork_output_path(output_dir, output_name, fallback_stem, overwrite):
    hypernetwork_name = normalize_output_name(output_name, fallback_stem)
    output_path = os.path.join(output_dir, f"{hypernetwork_name}.pt")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output hypernetwork already exists: {output_path}")
    return output_path, hypernetwork_name


def save_hypernetwork_payload(output_path, payload):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)


def build_hypernetwork_report(hypernetwork_name, output_path, source_hypernetwork, payload):
    branch_dims = sorted(int(key) for key in payload.keys() if str(key).isdigit())
    lines = [
        f"saved_hypernetwork={hypernetwork_name}",
        f"output_path={output_path}",
        f"source_hypernetwork={source_hypernetwork}",
        f"target_branch_dims={branch_dims}",
        f"projection_strategy={payload.get('embed_converter_projection_strategy', 'unknown')}",
    ]
    report = "\n".join(lines)
    print(report)
    return report


NODE_CLASS_MAPPINGS = {
    "SD15ToSDXLEmbeddingConverter": SD15ToSDXLEmbeddingConverter,
    "SD15FolderToSDXLEmbeddingBatchConverter": SD15FolderToSDXLEmbeddingBatchConverter,
    "SD15ToSDXLHypernetworkConverter": SD15ToSDXLHypernetworkConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD15ToSDXLEmbeddingConverter": "SD1.5 To SDXL Embedding Converter",
    "SD15FolderToSDXLEmbeddingBatchConverter": "SD1.5 Folder To SDXL Embedding Batch Converter",
    "SD15ToSDXLHypernetworkConverter": "SD1.5 To SDXL Hypernetwork Converter",
}
