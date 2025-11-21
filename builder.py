import argparse
import os
import torch
import shutil

import onnx
from onnxscript.rewriter import ort_fusions
from transformers import Qwen2_5_VLConfig, AutoModel
from torch.onnx._internal.exporter import _testing

import onnxscript
import onnx_ir as ir
from typing import Sequence
import onnx_ir.passes.common as common_passes


def _replace_functions(
    irmodel: ir.Model, attention_implementation: str = "LoopAttention"
) -> ir.Model:
    """A utility function to replace custom operations in a model with their expansions:
    Args:
        model: An ONNX ModelProto possibly containing calls to custom operations.
        attention_implementation: The attention implementation to use. Currently only "LoopAttention" and "PackedMultiHeadAttention" are supported.

    Returns:
        An updated ModelProto with custom operations replaced by their expansions.
    """

    custom = onnxscript.values.Opset("custom", 1)
    op = onnxscript.opset22
    msft_op = onnxscript.values.Opset("com.microsoft", 1)

    if attention_implementation == "LoopAttention":

        @onnxscript.script(opset=custom)
        def PackedAttention(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            scale: float,
            num_heads: int,
        ):
            # Shapes of input Q/K/V: [B, num_heads, seq_len, head_dim]

            # Convert Q/K/V to shape [B, seq_len, num_heads*head_dim]
            to_3d_shape = op.Constant(value_ints=[0, 0, -1])
            query_transposed = op.Transpose(query_states, perm=[0, 2, 1, 3])
            output_shape = op.Shape(query_transposed)
            query_3d = op.Reshape(query_transposed, to_3d_shape)
            value_3d = op.Reshape(
                op.Transpose(value_states, perm=[0, 2, 1, 3]), to_3d_shape
            )
            key_3d = op.Reshape(
                op.Transpose(key_states, perm=[0, 2, 1, 3]), to_3d_shape
            )

            num_patches = op.Size(cu_seqlens) - 1
            seq_axis = op.Constant(value_ints=[1])
            seq_axis_int32 = op.Cast(seq_axis, to=onnx.TensorProto.INT32)
            attn_output = op.Slice(
                value_3d, [0], [0], seq_axis
            )  # Initialize empty output
            for i in range(num_patches):
                i_1d = op.Reshape(i, [1])
                i_plus_1_1d = i_1d + 1
                start = op.Gather(cu_seqlens, i_1d, axis=0)
                end = op.Gather(cu_seqlens, i_plus_1_1d, axis=0)

                query_i = op.Slice(query_3d, start, end, seq_axis_int32)
                key_i = op.Slice(key_3d, start, end, seq_axis_int32)
                value_i = op.Slice(value_3d, start, end, seq_axis_int32)

                mha_output = msft_op.MultiHeadAttention(
                    query_i,
                    key_i,
                    value_i,
                    num_heads=num_heads,
                    scale=scale,
                )
                attn_output = op.Concat(attn_output, mha_output, axis=1)
            attn_output_4d = op.Reshape(attn_output, output_shape)
            return attn_output_4d  # [B, seq_len, num_heads, head_dim]

    elif attention_implementation == "PackedMultiHeadAttention":

        @onnxscript.script(opset=custom)
        def PackedAttention(
            query, key, value, cu_seqlens, scale: float, num_heads: int
        ):
            # Shapes of input Q/K/V: [B=1, num_heads, seq_len, head_dim]
            num_patches = op.Cast(op.Size(cu_seqlens), to=onnx.TensorProto.INT32) - 1
            # Identify lengths of each patch and max length
            starts = op.Slice(cu_seqlens, [0], [-1], [0])  # [num_patches]
            ends = op.Slice(
                cu_seqlens, [1], [9223372036854775807], [0]
            )  # [num_patches]
            lengths = ends - starts  # [num_patches]
            max_length = op.ReduceMax(lengths, [0], keepdims=0)  # [1]
            # Create token_offset required by the PackedMultiHeadAttention op
            # First create matrix: [
            #    [0, 1, 2, ..., max_length-1],
            #    [max_length, max_length+1, ..., 2*max_length-1],
            #    ... ]
            # zero_int64 = op.Constant(value_int=0)
            # zero_int32 = op.Cast(zero_int64, to=6)
            # one_int64 = op.Constant(value_int=0)
            # one_int32 = op.Cast(one_int64, to=6)
            rows = op.Range(0, num_patches, 1)  # [num_patches]
            rows_2d = op.Unsqueeze(rows, [1])  # [num_patches, 1]
            cols = op.Range(0, max_length, 1)  # [max_length]
            cols_2d = op.Unsqueeze(cols, [0])  # [1, max_length]
            position_matrix = (
                rows_2d * max_length + cols_2d
            )  # [num_patches, max_length]
            position_matrix_shape = op.Shape(position_matrix)
            # Now find positions of valid tokens and padding tokens
            # Position at column j in row i is valid if j < lengths[i]
            token_mask = cols_2d < op.Unsqueeze(
                lengths, [1]
            )  # [num_patches, max_length]
            token_mask_1d = op.Reshape(token_mask, [-1])  # [num_patches * max_length]
            # All other positions are padding
            padded_mask_1d = op.Not(token_mask_1d)
            valid_token_positions = op.Compress(
                position_matrix, token_mask
            )  # [total_valid_tokens]
            padded_token_positions = op.Compress(
                position_matrix, padded_mask_1d
            )  # [total_padded_tokens]
            token_offset_1d = op.Concat(
                valid_token_positions, padded_token_positions, axis=0
            )  # [num_patches * max_length]
            token_offset = op.Reshape(
                token_offset_1d, position_matrix_shape
            )  # [num_patches, max_length]

            # Convert query/key/value to shape (seq_len, num_heads* head_dim)
            # squeeze(0) => transpose(0,1) => reshape([0, -1])
            query_3d = op.Transpose(op.Squeeze(query, [0]), perm=[1, 0, 2])
            shape_3d = op.Shape(query_3d)
            query_2d = op.Reshape(query_3d, [0, -1])
            key_2d = op.Reshape(
                op.Transpose(op.Squeeze(key, [0]), perm=[1, 0, 2]), [0, -1]
            )
            value_2d = op.Reshape(
                op.Transpose(op.Squeeze(value, [0]), perm=[1, 0, 2]), [0, -1]
            )

            packed_attn_output_2d = msft_op.PackedMultiHeadAttention(
                query_2d,
                key_2d,
                value_2d,
                None,
                token_offset,
                cu_seqlens,
                scale=scale,
                num_heads=num_heads,
            )
            packed_attn_output_3d = op.Reshape(packed_attn_output_2d, shape_3d)
            return op.Unsqueeze(
                packed_attn_output_3d, [0]
            )  # [B, seq_len, num_heads, head_dim]

    else:
        raise ValueError(
            f"Unsupported attention implementation: {attention_implementation}. Supported implementations are 'LoopAttention' and 'PackedMultiHeadAttention'."
        )
    # Update the functions into the model
    irfunctions: list[ir.Function] = [
        ir.from_proto(PackedAttention.to_function_proto())
    ]
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
    pixel_values = pixel_values.to(args.precision).to(
        args.execution_provider.replace("dml", "cuda")
    )

    grid_thw = torch.tensor([[1, 98, 146]], dtype=torch.int64).to(
        args.execution_provider.replace("dml", "cuda")
    )

    # Dynamo export
    dummy_inputs = {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
    dynamic_shapes = {"pixel_values": {0: "num_patches"}, "image_grid_thw": None}

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
            report=True
        )

    # apply ort_fusions
    vision_onnx_program.model, optimized_count = ort_fusions.optimize_for_ort(
        vision_onnx_program.model
    )
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
    vision_onnx_program.model = _replace_functions(
        vision_onnx_program.model, args.attention_implementation
    )

    # NOTE: We need to rename output shape name to match the original name
    vision_onnx_program.model.graph.outputs[0].shape[0] = "num_logical_patches"

    # Save the ONNX model
    filename = "qwen2_5_vl-vision.onnx"
    vision_loop_export = os.path.join(args.output, "vision_loop_export")
    os.makedirs(vision_loop_export, exist_ok=True)
    vision_path = os.path.join(vision_loop_export, filename)
    vision_onnx_program.save(vision_path, external_data=True)
    # remove the intermediate folder
    shutil.rmtree(vision_init_export)

    # We need to compare to eager becasue the exported model contains custom ops
    onnx_outputs = vision_onnx_program(pixel_values, grid_thw)
    pytorch_outputs = model.eval().get_image_features(
        pixel_values=pixel_values, image_grid_thw=grid_thw
    )    

    torch.testing.assert_close(
        tuple(onnx_outputs),
        tuple(pytorch_outputs),
        atol=0.001,
        rtol=0.001,
        equal_nan=True,
        check_device=False,
    )
    

def build_embedding(args):
    # Dynamo export
    # assume 2 batches, each with 1 image input (3577 logical patches)
    batch_size, sequence_length, patches_per_image, out_hidden_size = (
        2,
        3606,
        3577,
        3584,
    )
    num_logical_patches = batch_size * patches_per_image
    inputs = {
        "input_ids": torch.randint(
            low=0,
            high=config.image_token_id,
            size=(batch_size, sequence_length),
            device=args.execution_provider.replace("dml", "cuda"),
            dtype=torch.int64,
        ),
        "image_features": torch.randn(
            num_logical_patches,
            out_hidden_size,
            device=args.execution_provider.replace("dml", "cuda"),
            dtype=args.precision,
        ),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image  # 3 + 3577 = 3580

    # Fill in with image token index
    inputs["input_ids"][0][2] = config.bos_token_id  # <start_of_image>
    inputs["input_ids"][0][
        img_start_index:img_end_index
    ] = config.image_token_id  # <image>
    inputs["input_ids"][0][img_end_index] = config.eos_token_id  # <end_of_image>

    inputs["input_ids"][1][2] = config.bos_token_id  # <start_of_image>
    inputs["input_ids"][1][
        img_start_index:img_end_index
    ] = config.image_token_id  # <image>
    inputs["input_ids"][1][img_end_index] = config.eos_token_id  # <end_of_image>

    dummy_inputs = (
        inputs["input_ids"],  # input_ids: torch.LongTensor
        inputs["image_features"],  # image_features: Optional[torch.FloatTensor] = None,
    )
    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
    }

    # NOTE: hack to embedding model export
    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )

    with torch.no_grad():
        embedding_onnx_program = torch.onnx.export(
            model,
            dummy_inputs,
            input_names=["input_ids", "image_features"],
            output_names=["inputs_embeds"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            optimize=True,
            opset_version=22,
        )
    # Test the parity of the exported model
    # _testing.assert_onnx_program(embedding_onnx_program)

    # Restore original forward method
    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )

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
        default=os.path.join(".", "cache_dir"),
        help="Cache directory for Hugging Face files and temporary ONNX external data files",
    )
    parser.add_argument(
        "--part",
        required=False,
        default="all",
        help="embedding, vision",
    )
    parser.add_argument(
        "-a",
        "--attention_implementation",
        required=False,
        default="LoopAttention",
        help="Attention implementation to use: LoopAttention, PackedMultiHeadAttention",
    )
    parser.add_argument(
        "--no_weights",
        action="store_true",
        help="If set, do not load model weights",
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

    if args.no_weights:
        # NOTE: Build model config without loading weights
        #       Feel free to adjust model config as needed
        config = Qwen2_5_VLConfig(
                vision_config={
                    "depth": 1,  # Only 1 vision layer instead of default 32
                    # "fullatt_block_indexes": [1],

                },
                text_config={
                    "num_hidden_layers": 4,
                    "max_window_layers": 4,  # Adjust this too since default is 80
                }
        )
        # Initialize model with random weights (no from_pretrained)
        model = AutoModel.from_config(
            config,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=args.precision,
        ).to(args.execution_provider.replace("dml", "cuda"))
    else:
        config = Qwen2_5_VLConfig.from_pretrained(args.input)
        model = AutoModel.from_pretrained(
            args.input,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=args.precision,
        ).to(args.execution_provider.replace("dml", "cuda"))

    # Build model components
    if args.part == "embedding":
        build_embedding(args)
    elif args.part == "vision":
        build_vision(args)
    else:
        build_embedding(args)
        build_vision(args)
