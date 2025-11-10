import argparse
import os
import subprocess
import sys
import torch

from onnxruntime_genai.models.builder import create_model
from transformers import Qwen2_5_VLConfig, AutoModel


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

    # Restore original forward method
    model.get_image_features, model.forward = model.forward, model.get_image_features

    # Save the ONNX model
    filename = "qwen-2_5-vision.onnx"
    vision_init_export = os.path.join(args.output, "vision_init_export")
    os.makedirs(vision_init_export, exist_ok=True)
    vision_path = os.path.join(vision_init_export, filename)
    vision_onnx_program.save(vision_path, external_data=True)

    # TODO(titaiwang): Try ort_fusion
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

# TODO(titaiwang): Check embedding model export
def build_embedding(args):
    # Dynamo export
    batch_size, sequence_length, num_img_tokens, image_length = 2, 268, 2, 256
    inputs = {
        "input_ids": torch.randint(low=0, high=config.image_token_index, size=(batch_size, sequence_length), device=args.execution_provider.replace("dml", "cuda"), dtype=torch.int64),
        "image_features": torch.randn(num_img_tokens, image_length, config.text_config.hidden_size, device=args.execution_provider.replace("dml", "cuda"), dtype=args.precision),
    }
    
    # Fill in with image token index
    inputs["input_ids"][0][2] = config.boi_token_index  # <start_of_image>
    inputs["input_ids"][0][3:255] = config.image_token_index # <image>
    inputs["input_ids"][0][255] = config.eoi_token_index  # <end_of_image>
    
    inputs["input_ids"][1][2] = config.boi_token_index  # <start_of_image>
    inputs["input_ids"][1][3:255] = config.image_token_index # <image>
    inputs["input_ids"][1][255] = config.eoi_token_index  # <end_of_image>
    
    dummy_inputs = (
        inputs["input_ids"],      # input_ids: torch.LongTensor
        inputs["image_features"], # image_features: Optional[torch.FloatTensor] = None,
    )
    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_images", 1: "image_length"},
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

    # Restore original forward method
    model.get_fused_input_embeddings, model.forward = model.forward, model.get_fused_input_embeddings

    # Save the ONNX model
    filename = "gemma-3-embedding.onnx"
    fpath_1 = os.path.join(args.output, filename)
    embedding_onnx_program.save(fpath_1, external_data=True)

def build_text(args):
    # Create ONNX model
    model_name = None
    precision = "int4"
    extra_options = {
        "exclude_embeds": "true",
        "filename": "gemma-3-text.onnx",
    }
    if args.precision == torch.float32:
        extra_options["int4_accuracy_level"] = 4
    create_model(model_name, args.input, args.output, precision, args.execution_provider, args.cache_dir, **extra_options)


def build_apply_multimodal_rotary_pos_emb(args):
        import transformers
        apply_multimodal_rotary_pos_emb = (
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb
        )

        class Model(torch.nn.Module):
            def forward(self, q, k, cos, sin):
                return apply_multimodal_rotary_pos_emb(q, k, cos, sin, [16, 24, 24])

        dtype = getattr(torch, args.precision)
        inputs = (
            torch.rand((1, 16, 348, 128), dtype=dtype),
            torch.rand((1, 2, 348, 128), dtype=dtype),
            torch.rand((3, 1, 348, 128), dtype=dtype),
            torch.rand((3, 1, 348, 128), dtype=dtype),
        )
        model = Model()
        ds = (
            {0: "a", 1: "b", 2: "c"},
            {0: "a", 1: "e", 2: "c"},
            {2: "c"},
            {2: "c"},
        )
        epo = torch.onnx.export(model, inputs, dynamic_shapes=ds)
        epo.save("apply_multimodal_rotary_pos_emb.onnx")


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
        default="vision",
        help="embedding, vision, or multi",
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
    elif args.part == "multi":
        build_apply_multimodal_rotary_pos_emb()
    else:
        build_vision(args)
    # 
    # build_text(args)