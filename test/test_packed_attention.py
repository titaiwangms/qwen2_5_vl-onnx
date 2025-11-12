import onnxscript
import onnx
import onnx.inliner
import numpy as np
import onnxruntime as ort
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

op = onnxscript.opset22
msft_op = onnxscript.values.Opset("com.microsoft", 1)

num_heads = 2
head_dim = 256
num_patches = 4

@onnxscript.script()
def batched_attention2(query_states, key_states, value_states, cu_seqlens, scale: float, num_heads: int):
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

qkv_type = onnxscript.FLOAT["B", num_heads, "seq_len", head_dim]
cu_seqlens_type = onnxscript.INT64["num_patches + 1"]
output_type = onnxscript.FLOAT["B", "seq_len", num_heads * head_dim]

def make_model_proto():
    @onnxscript.script()
    def packed_attention_model(query: qkv_type, key: qkv_type, value: qkv_type, cu_seqlens: cu_seqlens_type) -> output_type:
        return batched_attention2(query, key, value, cu_seqlens, scale=0.125, num_heads=num_heads)
    proto = packed_attention_model.to_model_proto()
    print(onnx.printer.to_text(proto))
    proto = onnx.inliner.inline_local_functions(proto)
    print(onnx.printer.to_text(proto))
    return proto

def generate_test_inputs():
    """Generate random numpy arrays for testing the packed attention model."""
    batch_size = 1
    
    # Generate random patch lengths and calculate total sequence length
    patch_lengths = np.random.randint(5, 15, size=num_patches, dtype=np.int64)
    seq_len = patch_lengths.sum()
    
    # Generate cumulative sequence lengths: [num_patches + 1]
    cu_seqlens = np.zeros(num_patches + 1, dtype=np.int64)
    cu_seqlens[1:] = np.cumsum(patch_lengths)
    
    # Generate Q, K, V tensors: [B, num_heads, seq_len, head_dim]
    query = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    key = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    value = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    
    print(f"Generated inputs:")
    print(f"  query shape: {query.shape}")
    print(f"  key shape: {key.shape}")
    print(f"  value shape: {value.shape}")
    print(f"  cu_seqlens shape: {cu_seqlens.shape}, values: {cu_seqlens}")
    
    return {
        "query": query,
        "key": key,
        "value": value,
        "cu_seqlens": cu_seqlens
    }

def run_onnx_model(model_proto, inputs):
    """Create ONNX Runtime session and run inference."""
    # Save model to a temporary location
    model_path = "packed_attention_model.onnx"
    onnx.save(model_proto, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Create ONNX Runtime session
    session_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, session_options, providers=providers)
    
    # Run inference
    print("\nRunning inference...")
    outputs = session.run(None, inputs)
    
    print(f"\nInference completed!")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output dtype: {outputs[0].dtype}")
    print(f"Output sample (first 5 values): {outputs[0].flatten()[:5]}")
    
    return outputs

def attention_pytorch(query_states, key_states, value_states, cu_seqlens, scaling):
    attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [
        torch.split(tensor, lengths.tolist(), dim=2)
        for tensor in (query_states, key_states, value_states)
    ]

    attn_outputs = [
        attention_interface(
            None,
            q,
            k,
            v,
            attention_mask=None,
            scaling=scaling,
            dropout=0.0,
            is_causal=False,
        )[0]
        for q, k, v in zip(*splits)
    ]
    attn_output = torch.cat(attn_outputs, dim=1)
    return attn_output

def run_pytorch_model(inputs, scale=0.125):
    """Run PyTorch attention with the same inputs."""
    print("\n" + "="*60)
    print("Running PyTorch attention...")
    print("="*60)
    
    # Convert numpy arrays to PyTorch tensors
    query = torch.from_numpy(inputs["query"])
    key = torch.from_numpy(inputs["key"])
    value = torch.from_numpy(inputs["value"])
    cu_seqlens = torch.from_numpy(inputs["cu_seqlens"])
    
    # Run PyTorch attention
    with torch.no_grad():
        output = attention_pytorch(query, key, value, cu_seqlens, scaling=scale)
    
    # Convert output back to numpy
    output_np = output.numpy()
    
    print(f"PyTorch output shape: {output_np.shape}")
    print(f"PyTorch output dtype: {output_np.dtype}")
    print(f"PyTorch output sample (first 5 values): {output_np.flatten()[:5]}")
    
    return output_np

def compare_outputs(onnx_output, pytorch_output, rtol=1e-3, atol=1e-5):
    """Compare ONNX Runtime and PyTorch outputs."""
    print("\n" + "="*60)
    print("Comparing outputs...")
    print("="*60)
    
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Check if shapes match
    if onnx_output.shape != pytorch_output.shape:
        print("❌ Shapes do not match!")
        return False
    
    # Compute differences
    abs_diff = np.abs(onnx_output - pytorch_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    print(f"\nMax absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    
    # Check if outputs are close
    are_close = np.allclose(onnx_output, pytorch_output, rtol=rtol, atol=atol)
    
    if are_close:
        print(f"\n✅ Outputs match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"\n❌ Outputs do NOT match within tolerance (rtol={rtol}, atol={atol})")
        
        # Show some mismatched values
        mismatches = ~np.isclose(onnx_output, pytorch_output, rtol=rtol, atol=atol)
        num_mismatches = np.sum(mismatches)
        print(f"Number of mismatched elements: {num_mismatches} / {onnx_output.size}")
        print(f"Percentage mismatched: {100 * num_mismatches / onnx_output.size:.2f}%")
    
    return are_close



if __name__ == "__main__":
    model_proto = make_model_proto()
    
    # Generate test inputs
    inputs = generate_test_inputs()
    
    # Run the model with ONNX Runtime
    onnx_outputs = run_onnx_model(model_proto, inputs)
    
    # Run the model with PyTorch
    pytorch_output = run_pytorch_model(inputs, scale=0.125)
    
    # Compare outputs
    compare_outputs(onnx_outputs[0], pytorch_output)

