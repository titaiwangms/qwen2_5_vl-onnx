### Requirements

| Package     | Version  |
|-------------|----------|
| onnxscript  | nightly  |
| torch       | nightly  |
| onnx        | 1.19.1   |
| transformers| nightly (5.0.0/dev0)  |

### Goals
We need to provide 2 models and a graph of decomposition to mROPE

1. We only need to provide vision and embedding models
2. We need to provide decomposition to mROPE(`apply_multimodal_rotary_pos_emb`)

### Relaxed constraints

1. We can assume only one image at a time. GenAI will take care of the for loop (eg: looping the vision model)
2. GenAI should be able to support some of the VisionRotaryEmbedding processing - windows indexing.
3. Flash attention 2 is the branch we should take. However, it currently has `cu_seqlens` which is not supported by contrib op GQA. We might need to manually update QKV to condider global/window attention.

### Key rewrites

1. Removed for loop and lists in `get_window_index`, as grid_t should always be 1, as well as the image counts
2. Used a custom attention to bypass flash attention 2 for now (FIXME)
3. Rewrite `torch.unique_consecutive`