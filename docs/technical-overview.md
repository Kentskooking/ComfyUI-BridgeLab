# Technical Overview

## Package Layout

- `__init__.py` exports the ComfyUI node mappings.
- `nodes/sd15_to_sdxl_embedding.py` contains the three node classes and the main orchestration logic.
- `utils/alignment.py` fits and caches the CLIP-L to CLIP-G alignment.
- `utils/embedding_io.py` loads source embeddings and writes SDXL `.safetensors` outputs.
- `utils/validation.py` runs optional phrase-based validation against the selected SDXL checkpoint.
- `utils/hypernetwork_projection.py` performs the experimental SD 1.5 to SDXL hypernetwork projection.

## Embedding Conversion Pipeline

The single-file embedding converter follows this sequence:

1. Load the source embedding from `.safetensors` or legacy `.pt`.
2. Verify that the embedding contains CLIP-L vectors, has width `768`, and does not already contain CLIP-G vectors.
3. Load the target SDXL checkpoint token embedding tables for CLIP-L and CLIP-G.
4. Build or reuse a cached alignment transform for that checkpoint.
5. Optionally validate the transform with sampled CLIP Interrogator phrases.
6. Keep the source CLIP-L vectors unchanged and synthesize CLIP-G vectors by mapping the source CLIP-L vectors through the fitted transform.
7. Save the result as a `.safetensors` embedding with metadata describing the source, checkpoint, fit score, and validation scores.

The core mapping happens in `convert_embedding_bundle()`. It copies `clip_l` directly and computes `clip_g` with `alignment.map_vectors()`.

## Alignment Fitting

`utils/alignment.py` learns a checkpoint-specific transform between SDXL CLIP-L and CLIP-G token tables:

- The source table is the SDXL CLIP-L token embedding table.
- The target table is the SDXL CLIP-G token embedding table.
- The fit uses a scaled orthogonal rectangular Procrustes solve based on the thin SVD of the cross-covariance matrix.
- The resulting transform stores a matrix, source mean, target mean, scale value, fit cosine score, checkpoint name, and checkpoint fingerprint.

The cache key is derived from:

- alignment version
- absolute checkpoint path
- file size
- file modification time in nanoseconds

Cached transforms are stored under `cache/alignments/` as `.safetensors`.

## Phrase Validation

When validation is enabled, `utils/validation.py` loads the selected checkpoint CLIP encoders and compares:

- the true CLIP-G encoding of each phrase token sequence
- the CLIP-G encoding produced after replacing content tokens with mapped CLIP-L token vectors

The validation report exposes:

- `validation_hidden_cosine`
- `validation_pooled_cosine`
- `validation_phrase_count`

Phrase sampling comes from text files in:

`custom_nodes/eden_comfy_pipelines/clip_utils/data`

That dependency is external to this repo. If the phrase files are missing, validation will fail when enabled.

## Batch Conversion

The batch converter walks an input folder and converts all supported embedding files:

- supported extensions come from `folder_paths.supported_pt_extensions`
- recursive mode preserves subfolder structure
- output is written under `models/embeddings/converted_sd15-SDXL`
- directory names and output filenames are normalized to safe filesystem components
- existing outputs are skipped unless `overwrite` is enabled

The node report records converted, skipped, and failed items, with truncation after `100` entries per section.

## Hypernetwork Projection

The hypernetwork converter is explicitly experimental. It only supports legacy Automatic1111-style linear hypernetworks with:

- `activation_func == "linear"`
- no layer norm
- no dropout
- source branches for dimensions `320`, `640`, `768`, and `1280`

The converter:

1. Flattens each two-layer linear branch into a single affine transform.
2. Projects lower-dimensional branches upward with a tiled partial isometry.
3. Combines native and lifted branches for SDXL-sized outputs.
4. Re-encodes the resulting affine transform back into the legacy two-layer branch structure.

The produced payload adds SDXL-oriented metadata and writes a `.pt` hypernetwork.

## Output Metadata

Converted embeddings include metadata such as:

- `modelspec.architecture`
- `modelspec.title`
- `embed_converter.source_embedding`
- `embed_converter.target_checkpoint`
- `embed_converter.checkpoint_fingerprint`
- `embed_converter.fit_token_cosine`
- validation metrics when validation ran
- copied source metadata namespaced under `embed_converter.source_meta.*`

## Current Assumptions and Limits

- Source textual inversion embeddings must be SD 1.5 style CLIP-L embeddings with width `768`.
- Existing SDXL embeddings are rejected rather than modified.
- Validation depends on another custom node's phrase data.
- Hypernetwork support is narrow and should be treated as a best-effort conversion path, not a guaranteed faithful transfer.
