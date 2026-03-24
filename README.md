# ComfyUI-BridgeLab

This nodepack converts SD 1.5 embeddings and hypernetworks into usable SDXL versions.

## Included Nodes

- `SD1.5 To SDXL Embedding Converter`
- `SD1.5 Folder To SDXL Embedding Batch Converter`
- `SD1.5 To SDXL Hypernetwork Converter`

## What It Does

- Converts CLIP-L-only SD 1.5 textual inversion embeddings into SDXL `.safetensors` embeddings.
- Batch converts folders of embeddings into `models/embeddings/converted_sd15-SDXL`.
- Projects supported legacy SD 1.5 hypernetworks into SDXL-sized branches.

## Installation

- Place this repository inside `ComfyUI/custom_nodes`.
- Restart ComfyUI.

## Documentation

- [Technical Overview](docs/technical-overview.md)
- [Node Reference](docs/node-reference.md)

## Notes

- Alignment fitting is cached per SDXL checkpoint.
- Optional phrase validation uses bundled CLIP Interrogator phrase files from `data/clip_interrogator`.
- Bundled phrase files are attributed in `data/clip_interrogator/ATTRIBUTION.md`.
- Hypernetwork conversion is experimental and only supports classic linear payloads.

## License

MIT. See [LICENSE](LICENSE).
