import os
import time
import sys
import onnxscript
import onnx
import onnx.inliner
import numpy as np
import ml_dtypes
import pandas
import onnxruntime as ort
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

op = onnxscript.opset22
msft_op = onnxscript.values.Opset("com.microsoft", 1)

# onnxscript does not support local types
cu_seqlens_type = onnxscript.INT32["num_patches + 1"]
num_heads = 2
head_dim = 256
qkv_type = onnxscript.FLOAT["B", num_heads, "seq_len", head_dim]
output_type = onnxscript.FLOAT["B", "seq_len", num_heads, head_dim]
os.makedirs("dump", exist_ok=True)


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
    seq_axis_int32 = op.Cast(seq_axis, to=onnx.TensorProto.INT32)
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
    num_patches = op.Cast(op.Size(cu_seqlens), to=onnx.TensorProto.INT32) - 1
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

def make_loop_model_proto(dtype, scale: float, num_heads: int):
    to = {
        np.float32: onnx.TensorProto.FLOAT,
        np.float16: onnx.TensorProto.FLOAT16,
        ml_dtypes.bfloat16: onnx.TensorProto.BFLOAT16,
    }[dtype]

    @onnxscript.script()
    def packed_attention_model(query: qkv_type, key: qkv_type, value: qkv_type, cu_seqlens: cu_seqlens_type) -> output_type:
        return op.Cast(
            loop_attention(
                op.Cast(query, to=to),
                op.Cast(key, to=to),
                op.Cast(value, to=to),
                cu_seqlens,
                scale=scale,
                num_heads=num_heads
            ),
            to=onnx.TensorProto.FLOAT
        ) 

    proto = packed_attention_model.to_model_proto()
    # print(onnx.printer.to_text(proto))
    proto = onnx.inliner.inline_local_functions(proto)
    # print(onnx.printer.to_text(proto))
    return proto

def make_packed_mha_model_proto(dtype, scale: float, num_heads: int):
    to = {
        np.float32: onnx.TensorProto.FLOAT,
        np.float16: onnx.TensorProto.FLOAT16,
        ml_dtypes.bfloat16: onnx.TensorProto.BFLOAT16,
    }[dtype]

    @onnxscript.script()
    def packed_attention_model(query: qkv_type, key: qkv_type, value: qkv_type, cu_seqlens: cu_seqlens_type) -> output_type:
        return op.Cast(
            packed_attention(
                op.Cast(query, to=to),
                op.Cast(key, to=to),
                op.Cast(value, to=to),
                cu_seqlens,
                scale=scale,
                num_heads=num_heads
            ),
            to=onnx.TensorProto.FLOAT
        ) 

    proto = packed_attention_model.to_model_proto()
    # print(onnx.printer.to_text(proto))
    proto = onnx.inliner.inline_local_functions(proto)
    # print(onnx.printer.to_text(proto))
    return proto

def generate_test_inputs(patch_size_min, patch_size_max, num_patches, num_heads, head_dim, np_dtype):
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
    
    print("Generated inputs:")
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

def run_onnx_model(model_proto, inputs, EP, exp_name, debug=False):
    """Create ONNX Runtime session and run inference."""
    # Save model to a temporary location
    dtype = inputs["query"].dtype
    model_path = f"dump/_test_packed_attention_model_{EP[0].replace('ExecutionProvider', '').lower()}_{str(dtype)}.{exp_name}.onnx"
    onnx.save(model_proto, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Create ONNX Runtime session
    session_options = ort.SessionOptions()
    session_options.optimized_model_filepath = f"{model_path}.opt.onnx"
    session_options.enable_profiling = True
    session_options.profile_file_prefix = f"{model_path}.profiling"
    if debug:
        session_options.log_severity_level = 0
        session_options.log_verbosity_level = 0
    session = ort.InferenceSession(model_path, session_options, providers=EP)
    
    # Run inference
    print("\nRunning inference...")
    dtype = inputs["query"].dtype
    inputs = {k: (v if v.dtype == np.int32 else v.astype(np.float32)) for k, v in inputs.items()}
    outputs = session.run(None, inputs)
    begin = time.perf_counter()
    for _ in range(20):
        outputs = session.run(None, inputs)
    duration = time.perf_counter() - begin
    outputs = [v.astype(dtype) for v in outputs]
    prof = session.end_profiling()

    # Anaylyse
    import matplotlib.pyplot as plt
    from onnx_diagnostic.helpers.rt_helper import js_profile_to_dataframe, plot_ort_profile
    df = js_profile_to_dataframe(prof, first_it_out=True)
    df.to_excel(f"{model_path}.prof.xlsx")
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    plot_ort_profile(df, ax[0], ax[1], model_path)
    fig.tight_layout()
    fig.savefig(f"{model_path}.png")
    
    print("\nInference completed!")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output dtype: {outputs[0].dtype}")
    print(f"Output sample (first 5 values): {outputs[0].flatten()[:5]}")
    
    return outputs, duration

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
    use_bfloat16 = inputs["query"].dtype == ml_dtypes.bfloat16
    
    # Convert numpy arrays to PyTorch tensors
    if use_bfloat16:
        query = torch.from_numpy(inputs["query"].astype(np.float32)).to(torch.bfloat16)
        key = torch.from_numpy(inputs["key"].astype(np.float32)).to(torch.bfloat16)
        value = torch.from_numpy(inputs["value"].astype(np.float32)).to(torch.bfloat16)
    else:
        query = torch.from_numpy(inputs["query"])
        key = torch.from_numpy(inputs["key"])
        value = torch.from_numpy(inputs["value"])
    cu_seqlens = torch.from_numpy(inputs["cu_seqlens"])
    
    # Run PyTorch attention
    with torch.no_grad():
        output = attention_pytorch(query, key, value, cu_seqlens, scaling=scale)
    
    # Convert output back to numpy
    if use_bfloat16:
        output_np = output.to(torch.float32).numpy().astype(ml_dtypes.bfloat16)
    else:
        output_np = output.numpy()
    
    print(f"PyTorch output shape: {output_np.shape}")
    print(f"PyTorch output dtype: {output_np.dtype}")
    print(f"PyTorch output sample (first 5 values): {output_np.flatten()[:5]}")
    
    return output_np

def compare_outputs(onnx_output, pytorch_output, mesg: str, rtol=1, atol=2e-3, ep=None):
    """Compare ONNX Runtime and PyTorch outputs."""
    print("\n" + "="*60)
    print("Comparing outputs...")
    print("="*60)
    
    print(f"ONNX {mesg} output shape: {onnx_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Check if shapes match
    if onnx_output.shape != pytorch_output.shape:
        print("❌ Shapes do not match!")
        return False
    
    # Compute differences
    abs_diff = np.abs(onnx_output - pytorch_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)
    
    max_abs_diff = np.max(abs_diff).astype(np.float32)
    max_rel_diff = np.max(rel_diff).astype(np.float32)
    mean_abs_diff = np.mean(abs_diff).astype(np.float32)

    print(f"\nMax absolute difference: {max_abs_diff:.6e}")
    print(f"Max relative difference: {max_rel_diff:.6e}")
    print(f"Mean absolute difference: {mean_abs_diff:.6e}")
    
    # Check if outputs are close
    are_close = np.allclose(onnx_output.astype(np.float32), pytorch_output.astype(np.float32), rtol=rtol, atol=atol)
    
    if are_close:
        print(f"\n✅ Outputs match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"\n❌ Outputs do NOT match within tolerance (rtol={rtol}, atol={atol})")
        
        # Show some mismatched values
        mismatches = ~np.isclose(onnx_output.astype(np.float32), pytorch_output.astype(np.float32), rtol=rtol, atol=atol)
        num_mismatches = np.sum(mismatches)
        print(f"Number of mismatched elements: {num_mismatches} / {onnx_output.size}")
        print(f"Percentage mismatched: {100 * num_mismatches / onnx_output.size:.2f}%")
    
    return dict(name=mesg, close=are_close, atol=atol, abs=max_abs_diff, rel=max_rel_diff, dtype=pytorch_output.dtype, ep=ep or "?")


if __name__ == "__main__":
    debug = "debug" in sys.argv
    only_bfloat16 = "bfloat16" in sys.argv
    choices = [
        (ml_dtypes.bfloat16, ["CUDAExecutionProvider"]),
        (ml_dtypes.bfloat16, ["CPUExecutionProvider"]),
        (np.float32, ["CUDAExecutionProvider"]),
        (np.float32, ["CPUExecutionProvider"]),
        (np.float16, ["CUDAExecutionProvider"]),
        (np.float16, ["CPUExecutionProvider"]),
    ]
    data = []

    for l_np_dtype, l_EP in choices:
        if only_bfloat16 and l_np_dtype != ml_dtypes.bfloat16:
            continue
        l_num_patches = 40
        l_patch_size_min = 20
        l_patch_size_max = 40
        l_scale = 0.125
        
        # Generate test inputs
        inputs = generate_test_inputs(l_patch_size_min, l_patch_size_max, l_num_patches, num_heads, head_dim, l_np_dtype)

        # Run the model with PyTorch
        pytorch_output = run_pytorch_model(inputs, scale=l_scale)
        
        # Run the loop-based model with ONNX Runtime
        model_proto = make_loop_model_proto(l_np_dtype, l_scale, num_heads)
        try:
            onnx_outputs, duration = run_onnx_model(model_proto, inputs, l_EP, "loopMHA")
            ok = True
        except ort.capi.onnxruntime_pybind11_state.NotImplemented as e:
            data.append(dict(name="Loop MHA", dtype=pytorch_output.dtype, ep=l_EP[0], ERR="NotImplemented", ERRMSG=str(e)))
            ok = False
            if debug:
                raise
        if ok:  
            obs = compare_outputs(onnx_outputs[0], pytorch_output, "Loop MHA", ep=l_EP[0])
            obs["duration"] = duration
            data.append(obs)

        # Run the packed_mha-based model with ONNX Runtime
        model_proto = make_packed_mha_model_proto(l_np_dtype, l_scale, num_heads)
        try:
            onnx_outputs, duration = run_onnx_model(model_proto, inputs, l_EP, "packedMHA", debug=debug)
            ok = True
        except ort.capi.onnxruntime_pybind11_state.NotImplemented as e:
            data.append(dict(name="Packed MHA", dtype=pytorch_output.dtype, ep=l_EP[0], ERR="NotImplemented", ERRMSG=str(e)))
            ok = False
            if debug:
                raise
        except ort.capi.onnxruntime_pybind11_state.InvalidGraph as e:
            data.append(dict(name="Packed MHA", dtype=pytorch_output.dtype, ep=l_EP[0], ERR="InvalidGraph", ERRMSG=str(e)))
            print("-------")
            print(e)
            print("-------")
            ok = False
            if debug:
                raise
        if ok:
            obs = compare_outputs(onnx_outputs[0], pytorch_output, "Packed MHA Attention", ep=l_EP[0])
            obs["duration"] = duration
            data.append(obs)

    # final result
    df = pandas.DataFrame(data)
    print(df)
    df.to_excel("dump/test_packed_attention.xlsx")

