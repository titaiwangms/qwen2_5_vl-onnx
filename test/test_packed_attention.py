import onnxscript
import onnx
import onnx.inliner
import numpy as np
import onnxruntime as ort
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

choice = 1  # Change to 1 to test FLOAT16
onnx_dtype, np_dtype, EP = [
    (onnxscript.FLOAT, np.float32, ["CPUExecutionProvider"]),
    (onnxscript.FLOAT16, np.float16, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
][choice]

op = onnxscript.opset22
msft_op = onnxscript.values.Opset("com.microsoft", 1)

num_heads = 2
head_dim = 256
num_patches = 40
patch_size_min = 20
patch_size_max = 40

@onnxscript.script()
def loop_attention(query_states, key_states, value_states, cu_seqlens, scale: float, num_heads: int):
    # Shapes of input Q/K/V: [B, num_heads, seq_len, head_dim]

    # Convert Q/K/V to shape [B, seq_len, num_heads*head_dim]
    to_3d_shape = op.Constant(value_ints=[0, 0, -1])
    query_transposed = op.Transpose(query_states, perm=[0, 2, 1, 3])
    output_shape = op.Shape(query_transposed)
    query_3d = op.Reshape(query_transposed, to_3d_shape)
    value_3d = op.Reshape(op.Transpose(value_states, perm=[0, 2, 1, 3]), to_3d_shape)
    key_3d = op.Reshape(op.Transpose(key_states, perm=[0, 2, 1, 3]), to_3d_shape)
    
    num_patches = op.Size(cu_seqlens) - 1
    seq_axis = op.Constant(value_ints=[1])
    seq_axis_int32 = op.Cast(seq_axis, to=6)
    attn_output = op.Slice(value_3d, [0], [0], seq_axis)  # Initialize empty output
    for i in range(num_patches):
        i_1d = op.Reshape(i, [1])
        i_plus_1_1d = i_1d + 1
        start = op.Gather(cu_seqlens, i_1d, axis=0)
        end = op.Gather(cu_seqlens, i_plus_1_1d, axis=0)

        query_i = op.Slice(query_3d, start, end, seq_axis_int32)
        key_i = op.Slice(key_3d, start, end, seq_axis_int32)
        value_i = op.Slice(value_3d, start, end, seq_axis_int32)

        mha_output = msft_op.MultiHeadAttention(
            query_i, key_i, value_i,
            num_heads=num_heads,
            scale=scale,
        )
        attn_output = op.Concat(attn_output, mha_output, axis=1)
    attn_output_4d = op.Reshape(attn_output, output_shape)
    return attn_output_4d  # [B, seq_len, num_heads, head_dim]

@onnxscript.script()
def packed_attention(query, key, value, cu_seqlens, scale: float, num_heads: int):
    # Shapes of input Q/K/V: [B=1, num_heads, seq_len, head_dim]
    num_patches = op.Cast(op.Size(cu_seqlens), to=6) - 1
    # Identify lengths of each patch and max length
    starts = op.Slice(cu_seqlens, [0], [-1], [0])  # [num_patches]
    ends = op.Slice(cu_seqlens, [1], [9223372036854775807], [0])  # [num_patches]
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
    position_matrix = rows_2d * max_length + cols_2d  # [num_patches, max_length]
    position_matrix_shape = op.Shape(position_matrix)
    # Now find positions of valid tokens and padding tokens
    # Position at column j in row i is valid if j < lengths[i]
    token_mask = cols_2d < op.Unsqueeze(lengths, [1])  # [num_patches, max_length]  
    token_mask_1d = op.Reshape(token_mask, [-1])  # [num_patches * max_length]
    # All other positions are padding
    padded_mask_1d = op.Not(token_mask_1d)
    valid_token_positions = op.Compress(position_matrix, token_mask)  # [total_valid_tokens]
    padded_token_positions = op.Compress(position_matrix, padded_mask_1d)  # [total_padded_tokens]
    token_offset_1d = op.Concat(valid_token_positions, padded_token_positions, axis=0)  # [num_patches * max_length]
    token_offset = op.Reshape(token_offset_1d, position_matrix_shape)  # [num_patches, max_length]

    # Convert query/key/value to shape (seq_len, num_heads* head_dim)
    # squeeze(0) => transpose(0,1) => reshape([0, -1])
    query_3d = op.Transpose(op.Squeeze(query, [0]), perm=[1,0,2])
    shape_3d = op.Shape(query_3d)
    query_2d = op.Reshape(query_3d, [0, -1])
    key_2d = op.Reshape(op.Transpose(op.Squeeze(key, [0]), perm=[1,0,2]), [0, -1])
    value_2d = op.Reshape(op.Transpose(op.Squeeze(value, [0]), perm=[1,0,2]), [0, -1])

    packed_attn_output_2d = msft_op.PackedMultiHeadAttention(
        query_2d, key_2d, value_2d, None, token_offset, cu_seqlens, scale=scale, num_heads=num_heads
    )
    packed_attn_output_3d = op.Reshape(packed_attn_output_2d, shape_3d)
    return op.Unsqueeze(packed_attn_output_3d, [0])  # [B, seq_len, num_heads, head_dim]

qkv_type = onnx_dtype["B", num_heads, "seq_len", head_dim]
cu_seqlens_type = onnxscript.INT32["num_patches + 1"]
output_type = onnx_dtype["B", "seq_len", num_heads, head_dim]

def make_model_proto():
    @onnxscript.script()
    def packed_attention_model(query: qkv_type, key: qkv_type, value: qkv_type, cu_seqlens: cu_seqlens_type) -> output_type:
        return loop_attention(query, key, value, cu_seqlens, scale=0.125, num_heads=num_heads)
    proto = packed_attention_model.to_model_proto()
    print(onnx.printer.to_text(proto))
    proto = onnx.inliner.inline_local_functions(proto)
    print(onnx.printer.to_text(proto))
    return proto

def generate_test_inputs():
    """Generate random numpy arrays for testing the packed attention model."""
    batch_size = 1
    
    # Generate random patch lengths and calculate total sequence length
    patch_lengths = np.random.randint(patch_size_min, patch_size_max, size=num_patches, dtype=np.int64)
    seq_len = patch_lengths.sum()
    
    # Generate cumulative sequence lengths: [num_patches + 1]
    cu_seqlens = np.zeros(num_patches + 1, dtype=np.int32)
    cu_seqlens[1:] = np.cumsum(patch_lengths)
    
    # Generate Q, K, V tensors: [B, num_heads, seq_len, head_dim]
    query = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)
    key = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)
    value = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np_dtype)
    
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
    session = ort.InferenceSession(model_path, session_options, providers=EP)
    
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

