"""
Use main branch of onnx-diagnostic.
Data saved with:

.. code-block:: python

    from onnx_diagnostic.helpers.mini_onnx_builder import (
        create_onnx_model_from_input_tensors,
    )
    import onnx

    proto = create_onnx_model_from_input_tensors(
        dict(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=(cu_seqlens[1:] - cu_seqlens[:-1]).max(),
            scaling=self.scaling,
            attn_output=attn_output,
        )
    )
    onnx.save(proto, "attention_inputs.onnx")
"""

import os
from onnx_diagnostic.helpers.mini_onnx_builder import (
    create_input_tensors_from_onnx_model,
)
from onnx_diagnostic.helpers import string_type


def get_attention_data():
    this = os.path.dirname(__file__)
    return create_input_tensors_from_onnx_model(
        os.path.join(this, "attention_inputs.onnx")
    )


def get_vision_forward():
    this = os.path.dirname(__file__)
    inputs = create_input_tensors_from_onnx_model(
        os.path.join(
            this, "Qwen2_5_VisionTransformerPretrainedModel.forward.inputs.onnx"
        )
    )
    outputs = create_input_tensors_from_onnx_model(
        os.path.join(
            this, "Qwen2_5_VisionTransformerPretrainedModel.forward.outputs.onnx"
        )
    )
    return dict(inputs=inputs, outputs=outputs)


def get_image_features():
    this = os.path.dirname(__file__)
    inputs = create_input_tensors_from_onnx_model(
        os.path.join(this, "get_image_features.inputs.onnx")
    )
    outputs = create_input_tensors_from_onnx_model(
        os.path.join(this, "get_image_features.outputs.onnx")
    )
    return dict(inputs=inputs, outputs=outputs)


print("-- attention")
data = get_attention_data()
for k, v in data.items():
    print(f"{k}: {string_type(v, with_shape=True, with_min_max=True)}")

print("-- vision_forward")
data = get_vision_forward()
for k, v in data.items():
    print(f"{k}: {string_type(v, with_shape=True, with_min_max=True)}")

print("-- get_image_features")
data = get_image_features()
for k, v in data.items():
    print(f"{k}: {string_type(v, with_shape=True, with_min_max=True)}")
