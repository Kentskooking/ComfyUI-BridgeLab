# Node Reference

## SD1.5 To SDXL Embedding Converter

Purpose:
Convert one SD 1.5 textual inversion embedding into an SDXL `.safetensors` embedding.

Inputs:

- `sd15_embedding`: source embedding from the ComfyUI `embeddings` model list
- `sdxl_checkpoint`: target SDXL checkpoint from the ComfyUI `checkpoints` model list
- `output_name`: optional output filename stem
- `overwrite`: allow replacing an existing output file
- `run_ci_validation`: run phrase-based validation after fitting the alignment
- `validation_phrase_count`: number of sampled phrases used during validation
- `validation_batch_size`: validation batch size

Returns:

- `embedding_name`
- `embedding_path`
- `fit_token_cosine`
- `validation_hidden_cosine`
- `validation_pooled_cosine`
- `report`

Behavior:

- Reads the source embedding through `load_embedding_bundle()`.
- Rejects anything that is missing CLIP-L, has a CLIP-L width other than `768`, or already contains CLIP-G.
- Builds or reuses a checkpoint-specific alignment cache.
- Saves the converted embedding into the active ComfyUI embeddings directory.

## SD1.5 Folder To SDXL Embedding Batch Converter

Purpose:
Convert every supported embedding in a folder into SDXL outputs.

Inputs:

- `input_folder_path`: source directory to scan
- `sdxl_checkpoint`: target SDXL checkpoint
- `name_suffix`: suffix appended to each output embedding stem
- `recursive`: include nested folders
- `overwrite`: allow replacing existing outputs
- `run_ci_validation`: run one shared alignment validation step before conversion
- `validation_phrase_count`: number of sampled phrases used during validation
- `validation_batch_size`: validation batch size

Returns:

- `output_directory`
- `converted_count`
- `skipped_count`
- `failed_count`
- `fit_token_cosine`
- `validation_hidden_cosine`
- `validation_pooled_cosine`
- `report`

Behavior:

- Writes outputs under `models/embeddings/converted_sd15-SDXL`.
- Reuses one alignment fit and one optional validation run for the whole batch.
- Preserves relative folder structure after sanitizing path components.
- Skips or fails items independently and records each outcome in the report.

## SD1.5 To SDXL Hypernetwork Converter

Purpose:
Project a supported legacy SD 1.5 hypernetwork into SDXL-sized branches.

Inputs:

- `sd15_hypernetwork`: source hypernetwork from the ComfyUI `hypernetworks` model list
- `output_name`: optional output filename stem
- `overwrite`: allow replacing an existing output file

Returns:

- `hypernetwork_name`
- `hypernetwork_path`
- `report`

Behavior:

- Loads the source payload onto CPU with `comfy.utils.load_torch_file()`.
- Converts only classic linear payloads accepted by `convert_linear_sd15_hypernetwork_to_sdxl()`.
- Writes the converted payload into the active ComfyUI hypernetworks directory as a `.pt` file.

## Common Failure Modes

- The selected checkpoint does not expose both SDXL CLIP-L and CLIP-G token embedding tables.
- The source embedding is already SDXL or is not a standard SD 1.5 textual inversion embedding.
- Validation is enabled but the bundled CLIP Interrogator phrase files are missing.
- The output file already exists and `overwrite` is disabled.
- The hypernetwork payload uses unsupported features such as non-linear activation, layer norm, or dropout.
