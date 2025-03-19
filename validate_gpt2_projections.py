import torch
from transformers import GPT2Model


def main():
    # Load GPT-2 model (small version)
    model = GPT2Model.from_pretrained('gpt2')
    hidden_size = model.config.hidden_size  # typically 768 for GPT-2
    print(f"GPT-2 hidden size: {hidden_size}")

    # Access the first transformer block's attention module
    # GPT2 uses a Conv1D layer for c_attn, which stores weight as (in_features, out_features)
    first_block = model.h[0]
    attn = first_block.attn
    c_attn = attn.c_attn
    print("\nc_attn linear layer:")
    print(f"Weight shape (original): {c_attn.weight.shape}")  # originally torch.Size([768, 2304])
    print(f"Bias shape: {c_attn.bias.shape}")      # expected shape: (2304,)

    # For a typical Linear layer, weights are (out_features, in_features).
    # Here, the Conv1D layer stores weights as (in_features, out_features), so we transpose.
    weight_transposed = c_attn.weight.transpose(0, 1)  # now shape is (2304, 768)
    print(f"Weight shape after transpose: {weight_transposed.shape}")

    # Validate that the transposed weight can be split into three equal parts
    if weight_transposed.shape[0] % 3 != 0:
        raise ValueError("The transposed c_attn weight's first dimension is not divisible by 3.")

    # Split the transposed weights along the first dimension
    weight_splits = torch.split(weight_transposed, hidden_size, dim=0)
    print("\nWeight splits (expected 3 projections: query, key, value):")
    for idx, w in enumerate(weight_splits):
        print(f"Projection {idx} weight shape: {w.shape}")

    # Similarly, for biases, no transpose is needed
    bias_splits = torch.split(c_attn.bias, hidden_size)
    print("\nBias splits:")
    for idx, b in enumerate(bias_splits):
        print(f"Projection {idx} bias shape: {b.shape}")

    # Demonstrate a forward pass with dummy input to inspect output splits
    dummy_input = torch.randn(1, 10, hidden_size)  # dummy input: batch_size=1, seq_len=10
    consolidated_output = c_attn(dummy_input)  # expected shape: (1, 10, 2304)
    print(f"\nConsolidated output shape: {consolidated_output.shape}")

    # Split the output into query, key, and value components along the last dimension
    output_splits = torch.split(consolidated_output, hidden_size, dim=-1)
    print("\nOutput splits:")
    for idx, out in enumerate(output_splits):
        print(f"Projection {idx} output shape: {out.shape}")

    # NEW SECTION: Directly use the full GPT2Attention module
    print("\n=== Testing full GPT2Attention module forward pass ===")
    
    # Prepare inputs for the attention module
    # For GPT2Attention, we need hidden_states as input
    batch_size = 1
    seq_length = 10
    hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    
    # We can optionally create an attention mask
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Call the full attention module's forward method
    # This should trigger the breakpoint in GPT2Attention.forward
    print("About to call the GPT2Attention forward method...")
    attn_outputs = attn(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=True
    )
    
    print("GPT2Attention forward method called successfully")
    attn_output = attn_outputs[0]  # First element is the attention output
    attention_weights = attn_outputs[1]  # Second element is the attention weights if output_attentions=True
    
    print(f"Attention output shape: {attn_output.shape}")
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")


if __name__ == '__main__':
    main() 