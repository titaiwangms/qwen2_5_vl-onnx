import argparse
import os
import subprocess
import sys
import torch

from onnxscript.rewriter import ort_fusions
from transformers import Qwen2_5_VLConfig, AutoModel
from torch.onnx._internal.exporter import _testing


def build_vision(args):
    # NOTE: Shape: [total_patches_across_all_images, patch_volume]
    # This is to accomodate to video input where multiple images are passed in a batch.


    if False:
        pixel_values = torch.randn((1292, 1176), dtype=torch.float32)
        # Scale the values to the range [-1, 0.95] to fit actual values we observed in the example.
        pixel_values = pixel_values * (0.95 - (-1)) + (-1)
        grid_thw = torch.tensor([[1, 34, 38]], dtype=torch.int64)
    else:
        from onnx_diagnostic.helpers.mini_onnx_builder import (
            create_input_tensors_from_onnx_model,
        )
        inputs = create_input_tensors_from_onnx_model("data/get_image_features.inputs.onnx")
        pixel_values = inputs["pixel_values"].to(torch.float32)
        grid_thw = inputs["image_grid_thw"]

    pixel_values = pixel_values.to(args.precision).to(args.execution_provider.replace("dml", "cuda"))
    grid_thw = grid_thw.to(args.execution_provider.replace("dml", "cuda"))

    # Dynamo export
    dummy_inputs = {
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw
    }
    dynamic_shapes = {
        "pixel_values": {0: "num_patches"},
        "image_grid_thw": {}
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

    _testing.assert_onnx_program(vision_onnx_program)
    
    # Restore original forward method
    model.get_image_features, model.forward = model.forward, model.get_image_features

    # Save the ONNX model
    filename = "qwen2_5_vl-vision.onnx"
    vision_init_export = os.path.join(args.output, "vision_init_export")
    os.makedirs(vision_init_export, exist_ok=True)
    vision_path = os.path.join(vision_init_export, filename)
    vision_onnx_program.save(vision_path, external_data=True)

    # ORT transformer optimizer
    vision_after_opt = os.path.join(args.output, "vision_after_opt")
    vision_opt_path = os.path.join(vision_after_opt, filename)
    subprocess.run(
        [
            f"{sys.executable}", "-m", "onnxruntime.transformers.optimizer",
            "--input", vision_path,
            "--output", vision_opt_path,
            "--model_type", "clip",
            "--num_heads", str(16),
            "--hidden_size", str(1280),
            "--use_external_data_format",
            "--opt_level", str(0),
            "--disable_shape_inference",
        ]
    )
    # shutil.rmtree(vision_init_export)

    # ORT 4-bits quantizer
    vision_final_path = os.path.join(args.output, filename)
    cmd = [
        f"{sys.executable}", "-m", "onnxruntime.quantization.matmul_nbits_quantizer",
        "--input_model", vision_opt_path,
        "--output_model", vision_final_path,
        "--block_size", str(32),
    ]
    if args.precision == torch.float32:
        cmd.extend(["--accuracy_level", str(4)])
    subprocess.run(cmd)
    # shutil.rmtree(vision_after_opt)

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
            opset_version=22,
        )

    _testing.assert_onnx_program(embedding_onnx_program)

    # Restore original forward method
    model.get_fused_input_embeddings, model.forward = model.forward, model.get_fused_input_embeddings

    # Save the ONNX model
    os.makedirs(args.output, exist_ok=True)
    filename = "qwen2_5_vl-embedding.onnx"
    fpath_1 = os.path.join(args.output, filename)
    embedding_onnx_program.save(fpath_1, external_data=True)

def build_mrope(args):
        import transformers
        apply_multimodal_rotary_pos_emb = (
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb
        )

        class Model(torch.nn.Module):
            def forward(self, q, k, cos, sin):
                return apply_multimodal_rotary_pos_emb(q, k, cos, sin, [16, 24, 24])
        dtype = args.precision
        inputs = (
            torch.rand((1, 28, 3606, 128), dtype=dtype),  # q
            torch.rand((1, 4, 3606, 128), dtype=dtype),  # k
            torch.rand((3, 1, 3606, 128), dtype=dtype),  # cos
            torch.rand((3, 1, 3606, 128), dtype=dtype),  # sin
        )
        model = Model()
        ds = (
            {0: "batch_size", 2: "seq_length"},  # q
            {0: "batch_size", 2: "seq_length"},  # k
            {1: "batch_size", 2: "seq_length"},  # cos
            {1: "batch_size", 2: "seq_length"},  # sin
        )
        epo = torch.onnx.export(model, inputs, dynamic_shapes=ds)
        _testing.assert_onnx_program(epo)
        epo.save("mrope.onnx")


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
        help="embedding, vision, or mrope",
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
    
    if args.part == "mrope":
        build_mrope(args)
    else:
        config = Qwen2_5_VLConfig.from_pretrained(args.input)
        model = AutoModel.from_pretrained(args.input, attn_implementation="sdpa", trust_remote_code=True, torch_dtype=args.precision).to(args.execution_provider.replace("dml", "cuda"))
        
        # Build model components
        if args.part == "embedding":
            build_embedding(args)
        elif args.part == "vision":
            # cp modeling_code/modeling_qwen2_5_vl.py ../transformers/src/transformers/models/qwen2_5_vl/
            # CUDA_VISIBLE_DEVICES=4,5,6 python builder.py -i Qwen/Qwen2.5-VL-7B-Instruct -o qwen_25_vl_bf16_vision -p bf16 -e cuda --part vision
            build_vision(args)
        else:
            build_embedding(args)
            build_vision(args)