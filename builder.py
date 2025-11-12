import argparse
import os 
import torch

from onnxscript.rewriter import ort_fusions
from transformers import Qwen2_5_VLConfig, AutoModel
from torch.onnx._internal.exporter import _testing

import onnxscript
import onnx_ir as ir
from typing import Sequence
import onnx_ir.passes.common as common_passes

def _replace_functions(irmodel: ir.Model, irfunctions: Sequence[ir.Function]) -> ir.Model:
    """A utility function to replace custom operations in a model with their expansions:
    Args:
        model: An ONNX ModelProto possibly containing calls to custom operations.
        functions: A sequence of FunctionProto defining the expansions for the custom operations.

    Returns:
        An updated ModelProto with custom operations replaced by their expansions.
    """
    model_functions = irmodel.functions
    if len(model_functions) != 0:
        # Since we use inlining, check that there are no model-local functions.
        raise ValueError("Input model cannot have model-local functions.")
    for func in irfunctions:
        model_functions[func.identifier()] = func
    # TODO (rama): Ideally, we should provide users more control over renaming strategy for inlined values.
    common_passes.InlinePass()(irmodel)
    common_passes.RemoveUnusedOpsetsPass()(irmodel)
    return irmodel

def build_vision(args):
    # NOTE: Shape: [total_patches_across_all_images, patch_volume]
    # This is to accomodate to video input where multiple images are passed in a batch.
    pixel_values = torch.randn((14308, 1176), dtype=torch.float32)
    # Scale the values to the range [-1, 0.95] to fit actual values we observed in the example.
    pixel_values = pixel_values * (0.95 - (-1)) + (-1)
    pixel_values = pixel_values.to(args.precision).to(args.execution_provider.replace("dml", "cuda"))

    grid_thw = torch.tensor([[1, 98, 146]], dtype=torch.int64).to(args.execution_provider.replace("dml", "cuda"))

    # Dynamo export
    dummy_inputs = {
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw
    }
    dynamic_shapes = {
        "pixel_values": {0: "num_patches"},
        "image_grid_thw": None
    }

    # NOTE: hack to image model export
    model.forward, model.get_image_features = model.get_image_features, model.forward

    with torch.no_grad():
        vision_onnx_program = torch.onnx.export(
            model,
            kwargs=dummy_inputs,
            input_names=["pixel_values", "image_grid_thw"],
            output_names=["image_features"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            optimize=True,
            opset_version=22,
        )
    
    # apply ort_fusions
    vision_onnx_program.model, optimized_count = ort_fusions.optimize_for_ort(vision_onnx_program.model)
    print("ORT optimized fusion counts:", optimized_count)
    
    # Restore original forward method
    model.get_image_features, model.forward = model.forward, model.get_image_features

    # Save the ONNX model
    filename = "qwen2_5_vl-vision.onnx"
    vision_init_export = os.path.join(args.output, "vision_init_export")
    os.makedirs(vision_init_export, exist_ok=True)
    vision_path = os.path.join(vision_init_export, filename)
    vision_onnx_program.save(vision_path, external_data=True)

    # graph surguery to change custom attention operator to onnxscript function
    custom = onnxscript.values.Opset("custom", 1)
    op = onnxscript.opset22
    msft_op = onnxscript.values.Opset("com.microsoft", 1)
    @onnxscript.script(opset=custom)
    def BatchedAttention(query_states, key_states, value_states, cu_seqlens, scale: float, num_heads: int):
        # Shapes of input Q/K/V: [B, num_heads, seq_len, head_dim]
    
        # Convert Q/K/V to shape [B, seq_len, num_heads*head_dim]
        to_3d_shape = op.Constant(value_ints=[0, 0, -1])
        query_3d = op.Reshape(op.Transpose(query_states, perm=[0, 2, 1, 3]), to_3d_shape)
        value_3d = op.Reshape(op.Transpose(value_states, perm=[0, 2, 1, 3]), to_3d_shape)
        key_3d = op.Reshape(op.Transpose(key_states, perm=[0, 2, 1, 3]), to_3d_shape)
    
        num_patches = op.Size(cu_seqlens) - 1
        seq_axis = op.Constant(value_ints=[1])
        attn_output = op.Slice(value_3d, [0], [0], seq_axis)  # Initialize empty output
        for i in range(num_patches):
            i_1d = op.Reshape(i, [1])
            i_plus_1_1d = i_1d + 1
            start = op.Gather(cu_seqlens, i_1d, axis=0)
            end = op.Gather(cu_seqlens, i_plus_1_1d, axis=0)
    
            query_i = op.Slice(query_3d, start, end, seq_axis)
            key_i = op.Slice(key_3d, start, end, seq_axis)
            value_i = op.Slice(value_3d, start, end, seq_axis)
    
            mha_output = msft_op.MultiHeadAttention(
                query_i, key_i, value_i,
                num_heads=num_heads,
                scale=scale,
            )
            attn_output = op.Concat(attn_output, mha_output, axis=1)
        return attn_output  # [B, seq_len, num_heads*head_dim]
    
    # Update the functions into the model
    functions = [ir.from_proto(BatchedAttention.to_function_proto())]
    vision_onnx_program.model = _replace_functions(vision_onnx_program.model, functions)

    # Save the ONNX model
    filename = "qwen2_5_vl-vision.onnx"
    vision_init_export = os.path.join(args.output, "vision_loop_export")
    os.makedirs(vision_init_export, exist_ok=True)
    vision_path = os.path.join(vision_init_export, filename)
    vision_onnx_program.save(vision_path, external_data=True)

    _testing.assert_onnx_program(vision_onnx_program)
    
    # op-level verification
    # from torch.onnx._internal.exporter import _verification
    # v_info = _verification.verify_onnx_program(vision_onnx_program, kwargs=dummy_inputs)


    # TODO(titaiwang): We probably need to change output dimension name for image_features
    # to match embedding model input.

def build_embedding(args):
    # Dynamo export
    # assume 2 batches, each with 1 image input (3577 logical patches)
    batch_size, sequence_length, patches_per_image, out_hidden_size = 2, 3606, 3577, 3584
    num_logical_patches = batch_size * patches_per_image
    inputs = {
        "input_ids": torch.randint(low=0, high=config.image_token_id, size=(batch_size, sequence_length), device=args.execution_provider.replace("dml", "cuda"), dtype=torch.int64),
        "image_features": torch.randn(num_logical_patches, out_hidden_size, device=args.execution_provider.replace("dml", "cuda"), dtype=args.precision),
    }
    
    img_start_index = 3
    img_end_index = img_start_index + patches_per_image # 3 + 3577 = 3580

    # Fill in with image token index
    inputs["input_ids"][0][2] = config.bos_token_id  # <start_of_image>
    inputs["input_ids"][0][img_start_index:img_end_index] = config.image_token_id # <image>
    inputs["input_ids"][0][img_end_index] = config.eos_token_id  # <end_of_image>

    inputs["input_ids"][1][2] = config.bos_token_id  # <start_of_image>
    inputs["input_ids"][1][img_start_index:img_end_index] = config.image_token_id # <image>
    inputs["input_ids"][1][img_end_index] = config.eos_token_id  # <end_of_image>

    dummy_inputs = (
        inputs["input_ids"],      # input_ids: torch.LongTensor
        inputs["image_features"], # image_features: Optional[torch.FloatTensor] = None,
    )
    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
    }
        
    # NOTE: hack to embedding model export
    model.get_fused_input_embeddings, model.forward = model.forward, model.get_fused_input_embeddings
    
    with torch.no_grad():
        embedding_onnx_program = torch.onnx.export(
            model,
            dummy_inputs,
            input_names=["input_ids", "image_features"],
            output_names=["inputs_embeds"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            optimize=True,
            opset_version=23,
        )
    # Test the parity of the exported model
    _testing.assert_onnx_program(embedding_onnx_program)

    # Restore original forward method
    model.get_fused_input_embeddings, model.forward = model.forward, model.get_fused_input_embeddings

    # Save the ONNX model
    os.makedirs(args.output, exist_ok=True)
    filename = "qwen2_5_vl-embedding.onnx"
    fpath_1 = os.path.join(args.output, filename)
    embedding_onnx_program.save(fpath_1, external_data=True)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to folder on disk containing the Hugging Face config, model, tokenizer, etc.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to folder to store ONNX model and additional files (e.g. GenAI config, external data files, etc.)",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        choices=["bf16", "fp16", "fp32"],
        help="Precision to export PyTorch components with",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=True,
        choices=["cpu", "cuda", "dml"],
        help="Execution provider",
    )

    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        default=os.path.join('.', 'cache_dir'),
        help="Cache directory for Hugging Face files and temporary ONNX external data files",
    )
    parser.add_argument(
        "--part",
        required=False,
        default="all",
        help="embedding, vision",
    )

    args = parser.parse_args()
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    args.precision = mapping[args.precision]
    return args


if __name__ == "__main__":
    args = get_args()
    

    config = Qwen2_5_VLConfig.from_pretrained(args.input)
    model = AutoModel.from_pretrained(args.input, attn_implementation="sdpa", trust_remote_code=True, torch_dtype=args.precision).to(args.execution_provider.replace("dml", "cuda"))
    
    # Build model components
    if args.part == "embedding":
        build_embedding(args)
    elif args.part == "vision":
        build_vision(args)
    else:
        build_embedding(args)
        build_vision(args)