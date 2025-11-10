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

### Key rewrites

1. Remove for loop and lists in `get_window_index`, as grid_t should always be 1