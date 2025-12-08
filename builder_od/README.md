# Export visual embedding of Qwen/Qwen2.5-VL-7B-Instruct

## requirements

```
git+https://github.com/sdpython/experimental-experiment.git
huggingface_hub>=1.2.1
onnx-diagnostic>=0.8.4
onnxruntime>=1.23
torch>=2.9  # weekly is better
transformers>=4.57
```

## Examples

```bash
python builder_od.py -m Qwen/Qwen2.5-VL-7B-Instruct --device cpu --dtype float32 --exporter onnx-dynamo --pretrained --second-input --zip
```

Cheat sheet for tar commmands. To make a tar:
``tar -czvf model.tar.gz model.onnx model.data``
And to untar:
``tar -xzvf model.tar.gz``.

## Reritings

* [overview](https://sdpython.github.io/doc/onnx-diagnostic/dev/status/patches_diff.html#auto-patch-transformers-qwen2-5-vlforconditionalgeneration-prepare-inputs-for-generation-patched-qwen2-5-vlforconditionalgeneration-prepare-inputs-for-generation)
* code: [_patch_transformers_qwen2_5.py](https://github.com/sdpython/onnx-diagnostic/blob/main/onnx_diagnostic/torch_export_patches/patches/_patch_transformers_qwen2_5.py)

## Attention

The attention is either implemented with ``MultiHeadAttention`` in a loop, either with ``PackedMultiHeadAttention``.
The choice is made based on the device. It is possible to overwrite this by by setting
environment variable to ``QWEN25ATTENTION`` to:

* ``PACKED``: PackedMultiHeadAttention
* ``LOOPMHA``: Loop over MultiHeadAttention
* ``LOOPA24``: Loop over Attention(24), needs opset 23 or 24.
