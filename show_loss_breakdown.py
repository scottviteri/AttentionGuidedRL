#!/usr/bin/env python3
"""
A simple script to demonstrate the loss breakdown in verbose mode.
This will run a single training step and show the breakdown of policy loss vs KL loss.
"""

import torch
from src.model import setup_model_and_tokenizer, apply_lora_adapter, create_model_copy
from src.data import iter_key_value_pairs, QKVStep
from src.embeddings import register_embedding_hook, extract_embeddings
from src.training import generate_query, compute_trajectory_rewards, Trajectory, train_step
from src.config import GENERATION_BATCH_SIZE, DEVICE, NUM_KV_PAIRS, INITIAL_PROMPT, KL_PENALTY_COEFFICIENT

def main():
    print("Setting up models and tokenizer...")
    base_model, adapter_model, tokenizer = setup_model_and_tokenizer()
    
    # Register embedding hooks
    embeddings_dict, hook_remover = register_embedding_hook(adapter_model)
    
    try:
        # Create optimizer
        optimizer = torch.optim.Adam(adapter_model.parameters(), lr=0.001)
        
        # Initialize reward stats
        reward_stats = {"mean": 0.0, "std": 1.0, "count": 10}  # Pretend we've seen 10 episodes already
        
        # Create a copy of the current model for KL divergence
        previous_model = create_model_copy(adapter_model)
        
        # Generate a batch of key-value pairs
        print("Generating key-value pairs...")
        batch_size = GENERATION_BATCH_SIZE
        kv_pair_generator = iter_key_value_pairs(
            batch_size=batch_size,
            embedding_fn=lambda x: extract_embeddings(adapter_model, x, embeddings_dict)
        )
        
        # Get a few key-value pairs
        qkv_steps = [next(kv_pair_generator) for _ in range(3)]
        
        # Create initial context
        device = next(adapter_model.parameters()).device
        initial_tokens = tokenizer(
            [INITIAL_PROMPT] * batch_size,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).input_ids.to(device)
        
        # Add query information to the QKV steps
        context_text = tokenizer.batch_decode(initial_tokens)
        for step in qkv_steps:
            # Generate a query for this step
            query_tokens = generate_query(
                adapter_model,
                tokenizer,
                context_text,
            )
            query_text = tokenizer.batch_decode(query_tokens)
            query_embeddings = extract_embeddings(adapter_model, query_tokens, embeddings_dict)
            
            # Update the step with the generated query information
            step.query_tokens = query_tokens
            step.query_text = query_text
            step.query_embedding = query_embeddings
        
        # Create a trajectory with the QKV steps
        trajectory = Trajectory(qkv_steps=qkv_steps)
        
        # Compute rewards
        print("Computing trajectory rewards...")
        compute_trajectory_rewards(
            trajectory,
            adapter_model,
            base_model,
            initial_tokens,
            tokenizer=tokenizer,
            verbose=True
        )
        
        # Now run a training step with verbose output to see the loss breakdown
        print("\nRunning training step with loss breakdown...")
        loss, num_filtered = train_step(
            trajectory,
            adapter_model,
            base_model,
            previous_model,
            optimizer,
            reward_stats,
            KL_PENALTY_COEFFICIENT,
            verbose=True
        )
        
        print(f"\nTraining step complete. Final loss: {loss:.4f}")
        
    finally:
        # Clean up
        hook_remover()

if __name__ == "__main__":
    main() 