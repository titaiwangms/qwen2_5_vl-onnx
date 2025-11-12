import os

print("-- import onnx")
import onnx
import onnx.helper as oh

print("-- import onnxruntime")
import onnxruntime

print("-- import torch")
import torch

print("-- import transformers")
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

print("-- import onnx-diagnostic")
from onnx_diagnostic.helpers.mini_onnx_builder import (
    create_input_tensors_from_onnx_model,
)
from onnx_diagnostic.helpers import string_type, max_diff

print("-- start")


def get_attention_data():
    this = os.path.dirname(__file__)
    return create_input_tensors_from_onnx_model(
        os.path.join(this, "attention_inputs.onnx")
    )


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


class BigMaskAttention(torch.nn.Module):
    def forward(self, query_states, key_states, value_states, cu_seqlens, scaling):
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        indices = torch.arange(
            cu_seqlens.max(), dtype=cu_seqlens.dtype, device=cu_seqlens.device
        )
        dot = (cu_seqlens.unsqueeze(1) <= indices.unsqueeze(0)).to(cu_seqlens.dtype)
        dot = dot.sum(dim=0)
        mask = dot.unsqueeze(1) - dot.unsqueeze(0)
        bool_mask = mask == 0
        bool_mask = bool_mask.unsqueeze(0).unsqueeze(0)

        torch._check(bool_mask.shape[2] == key_states.shape[2])
        torch._check(bool_mask.shape[3] == key_states.shape[2])

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=bool_mask,
            scaling=scaling,
            dropout=0.0,
            is_causal=False,
        )
        return attn_output


def _generate_packed_to_padded_mapping_2d(cu_seqlens):
    """
    Generate a tensor that maps packed sequence positions to their logical positions
    in a padded 2D tensor using 2D matrix approach with boolean masking.

    Args:
        cu_seqlens: Cumulative sequence lengths tensor, e.g., [0, 1, 3, 7]

    Returns:
        mapping_tensor: Tensor containing the mapping from packed to padded positions

    Example:
        cu_seqlens = [0, 1, 3, 7]  # lengths: [1, 2, 4]
        Returns: [0, 4, 5, 8, 9, 10, 11, 1, 2, 3, 6, 7]
    """
    device = cu_seqlens.device
    dtype = cu_seqlens.dtype

    # Get sequence lengths
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    num_sequences = lengths.shape[0]
    max_length = lengths.max()

    # Generate the entire 2D matrix of position indices
    # Shape: [num_sequences, max_length]
    # Example: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    row_indices = torch.arange(num_sequences, device=device, dtype=dtype).unsqueeze(1)
    col_indices = torch.arange(max_length, device=device, dtype=dtype).unsqueeze(0)
    position_matrix = row_indices * max_length + col_indices

    # Create boolean mask for unpadded positions
    # Shape: [num_sequences, max_length]
    # True where position is unpadded, False where padded
    unpadded_mask = col_indices < lengths.unsqueeze(1)

    # Extract unpadded positions (flatten and select)
    unpadded_positions = position_matrix[unpadded_mask]

    # Extract padded positions (flatten and select complement)
    padded_positions = position_matrix[~unpadded_mask]

    # Concatenate: unpadded positions first, then padded positions
    mapping_tensor = torch.cat([unpadded_positions, padded_positions])

    batch_size = cu_seqlens.shape[0] - 1
    mapping_tensor = mapping_tensor.reshape(batch_size, -1)

    return mapping_tensor


def packed_attention_onnx(query_states, key_states, value_states, cu_seqlens, scaling):
    """
    // Shapes of inputs and output:
    // When Q, K and V are not packed:
    //   Input 'query':                      (token_count, hidden_size)
    //   Input 'key':                        (token_count, hidden_size)
    //   Input 'value':                      (token_count, v_hidden_size)
    // When Q, K and V are packed:
    //   Input 'query':                      (token_count, num_heads, 3, head_size)
    //   Input 'key':                        None
    //   Input 'value':                      None
    // Input 'token_offset':                 (batch_size, sequence_length)
    // Input 'cumulative_sequence_length':   (batch_size + 1)
    // Input 'attention_bias':               (batch_size or 1, num_heads or 1, sequence_length, sequence_length) or None
    // Output 'output':                      (token_count, v_hidden_size)
    """
    itype = {
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.bfloat16: onnx.TensorProto.BFLOAT16,
        torch.float32: onnx.TensorProto.FLOAT,
    }[query_states.dtype]

    token_offset = _generate_packed_to_padded_mapping_2d(cu_seqlens)
    #packed_qkv = torch.stack(
    #    [query_states, key_states, value_states], dim=2  # Insert 3 to dim=2
    #)

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node(
                    "PackedMultiHeadAttention",
                    ["q", "k", "v", "", "offset", "cu_seqlens"],
                    ["attn"],
                    domain="com.microsoft",
                    scale=scaling,
                    num_heads=16,
                )
            ],
            "name",
            [
                oh.make_tensor_value_info("q", itype, ["batch", "a", "b", "c"]),
                oh.make_tensor_value_info("k", itype, ["batch", "a", "b", "c"]),
                oh.make_tensor_value_info("v", itype, ["batch", "a", "b", "c"]),
                # oh.make_tensor_value_info("packed", itype, ["a", "b", "c"]),
                oh.make_tensor_value_info("offset", onnx.TensorProto.INT32, ["l1"]),
                oh.make_tensor_value_info("cu_seqlens", onnx.TensorProto.INT32, ["l2"]),
            ],
            [
                oh.make_tensor_value_info("attn", itype, ["f"]),
            ],
        ),
        opset_imports=[oh.make_opsetid("", 22), oh.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    providers = [
        (
            "CUDAExecutionProvider"
            if "cuda" in str(query_states.device)
            else "CPUExecutionProvider"
        )
    ]
    feeds = dict(
        q=query_states.detach().cpu().numpy(),
        k=key_states.detach().cpu().numpy(),
        v=value_states.detach().cpu().numpy(),
        # packed=packed_qkv.detach().cpu().numpy(),
        offset=token_offset.to(torch.int32).detach().cpu().numpy(),
        cu_seqlens=cu_seqlens.to(torch.int32).detach().cpu().numpy(),
    )
    print(f"-- feeds: {string_type(feeds, with_shape=True)}")
    print(
        f"-- create ort session with providers {providers}, device={query_states.device!r}"
    )
    sess = onnxruntime.InferenceSession(model.SerializeToString(), providers=providers)
    return sess.run(None, feeds)[0]


print("-- attention")
data = get_attention_data()
for k, v in data.items():
    print(f"{k}: {string_type(v, with_shape=True, with_min_max=True)}")
if torch.cuda.is_available():
    print("-- move to cuda")
    data = {
        k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in data.items()
    }

expected = data["attn_output"]
query_states, key_states, value_states, cu_seqlens, scaling = (
    data["query_states"],
    data["key_states"],
    data["value_states"],
    data["cu_seqlens"],
    data["scaling"],
)

print("-- qwen implementation")
got = attention_pytorch(query_states, key_states, value_states, cu_seqlens, scaling)
diff = max_diff(expected, got)
print("-- qwen original implementation:", diff)

print("-- qwen implementation with float16")
got = attention_pytorch(
    query_states.to(torch.float16),
    key_states.to(torch.float16),
    value_states.to(torch.float16),
    cu_seqlens,
    scaling,
)
diff = max_diff(expected.to(torch.float16), got)
print("-- qwen original implementation with float16:", diff)

print("-- bigmask implementation")
got = BigMaskAttention()(query_states, key_states, value_states, cu_seqlens, scaling)
diff = max_diff(expected, got)
print("-- bigmask diff", diff)


print("-- packed mha ort implementation float16")
got = packed_attention_onnx(
    query_states.to(torch.float16),
    key_states.to(torch.float16),
    value_states.to(torch.float16),
    cu_seqlens,
    scaling,
)
diff = max_diff(expected.to(torch.float16), got)
print("-- packed mha ort fp16", diff)

print("-- packed mha ort implementation bfloat16")
got = packed_attention_onnx(query_states, key_states, value_states, cu_seqlens, scaling)
diff = max_diff(expected, got)
print("-- packed mha ort bf16", diff)
