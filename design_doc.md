# Attention-Guided Reinforcement Learning for Self-Directed Language Model Training

## Abstract

This project presents a reinforcement learning (RL)-based active learning framework that enables a base language model, **Llama-3.2-3B**, to autonomously guide its training by sequencing non-overlapping key–value pairs from Wikipedia articles. The model generates queries using a consistent "Query: _ Response: _" prompt format, with embeddings extracted from the last attention layer and averaged across the sequence length. These embeddings, processed with grouped query attention, compute multi-head aware similarity scores against pre-embedded keys. An RL agent orders the key–value pairs into the context window, receiving a reward based on the average conditional log probability of value tokens, normalized against the pre-trained model's baseline. The query-generation policy is refined using policy gradients with a KL divergence penalty, and updates are applied to high-reward trajectories filtered via a moving average. The implementation leverages LoRA for efficient training, synchronized token spaces without padding, functional dataloading with iterators, and extensive pytest testing for reliability.

---

## 1. Motivation

- **Bridging Natural Language and Active Learning:**  
  This approach uses Wikipedia's natural language data to exploit the model's linguistic capabilities for self-directed learning, moving beyond abstract active learning tasks.

- **Efficient Context Window Utilization:**  
  Non-overlapping key–value pairs are pre-embedded and dynamically ordered to maximize token efficiency, with each pair used exactly once per trajectory.

- **Self-Directed Curriculum via RL:**  
  The RL agent sequences training examples to optimize learning progress, measured by improvements in log probability, with filtering to prioritize high-quality updates.

- **Stable Training:**  
  A KL divergence penalty constrains shifts in the query generation distribution, ensuring training stability.

---

## 2. System Architecture

### 2.1. Components

- **Base Language Model:**  
  - *Model:* Llama-3.2-3B (base version, not fine-tuned).  
  - *Role:* Generates queries and predicts values for text completion.  
  - *Training:* Adapted using Low-Rank Adaptation (LoRA) with standard parameters (rank=8, alpha=16, dropout=0.05), with gradients enabled only for the adapter.  
  - *Clarification:* The configuration for grouped query attention (e.g., number of query heads, key-value groups, head dimension) can be inferred from the model using attributes such as `first_layer.self_attn.k_proj` and `first_layer.self_attn.q_proj`.

- **Key–Value Memory Database:**  
  - *Source:* Wikipedia articles, filtered to ensure each article has at least `tokens_per_kv_pair * num_kv_pairs` tokens, where `tokens_per_kv_pair = tokens_per_key + tokens_per_value` (e.g., 20 tokens per key + 20 tokens per value = 40 tokens per pair).  
  - *Keys:* Embeddings from the last attention layer of Llama-3.2-3B, averaged over sequence length.  
  - *Values:* Corresponding text segments.  
  - *Mechanism:* For each trajectory, a random subset (e.g., 1/4) of pairs is selected; chosen keys are removed to prevent reuse within the trajectory.

- **Query and Key Embeddings:**  
  - *Extraction:* Model hooks capture embeddings from the last attention layer during inference using `model.model.model.layers[-1].self_attn.q_proj` as the target module.  
  - *Processing:* Embeddings are averaged across the sequence length to produce a single vector per query or key.  
  - *Similarity:* A multi-head aware mechanism accounts for **grouped query attention**. After computing dot products per head-group pair and applying softmax per head, the probability distributions are averaged across the head dimension to obtain a single probability distribution over keys.  
  - *Clarification:* The baseline (counterfactual) query for reward normalization is generated with the same context and length as the trained query.

- **RL Agent:**  
  - *Action:* Sequences key–value pairs into the context, starting with the first key-value pair from the Wikipedia article as an in-context example.  
  - *Reward:* Average conditional log probability of value tokens, normalized by subtracting the pre-trained model's log probability of the value given a counterfactual query.  
  - *Optimization:* Policy gradients with KL regularization; updates use trajectories exceeding one standard deviation above the moving average reward.
  - *Termination:* Training runs for a fixed number of episodes (`num_episodes` parameter), with reward serving as the primary evaluation metric.

---

## 3. Methodology

### 3.1. Data Flow and Training Loop

1. **Initialization:**  
   - Load Llama-3.2-3B in `bfloat16` on CUDA (or CPU if unavailable).  
   - Load Wikipedia articles using a functional iterator-based dataloader, filtering articles with fewer than `tokens_per_kv_pair * num_kv_pairs` tokens.  
   - Pre-compute key embeddings from the data source.  
   - Initialize the RL policy with a LoRA adapter (rank=8, alpha=16, dropout=0.05).  
   - Use the first 15 trajectories as a warm-up to establish reward statistics.
   - Set learning rate for the BitsAndBytes 8-bit Adam optimizer to 2e-4.

2. **Trajectory Generation:**  
   - For each episode:  
     1. Start with the first key-value pair from the Wikipedia article included in the context.  
     2. While key–value pairs remain in the subset:  
        - Generate a query using the current model (base + LoRA) with the prompt ending in " Query:".  
        - Extract and average the query embedding from the last attention layer.  
        - Compute multi-head aware similarity with remaining key embeddings, averaging probabilities across heads.  
        - Apply softmax for a probability distribution over keys.  
        - Sample a key, remove it from the database, and append the full " Query: [query] Response: [response]" to the context.  
     3. During each step, compute the reward normalization using a counterfactual query generated by the pre-trained model.  
   - Record the trajectory (queries, keys, values, rewards).  
   - *Clarification:* The counterfactual query is generated by the pre-trained model with the same context and length as the trained query.

3. **Reward Computation:**  
   - Compute conditional log probability of value tokens given the context and query using the current model, including the " Response: " prefix.  
   - Normalize by subtracting the pre-trained model's log probability of the value given a counterfactual query.  
   - Average the normalized log probabilities across the trajectory for the reward.

4. **Trajectory Filtering:**  
   - Post-warm-up, maintain a moving average of the mean and standard deviation of rewards over the entire trajectory history (no fixed window, no weighting).  
   - Filter trajectories with rewards >1 SD above the moving average for updates.

5. **Policy Update:**  
   - Compute policy gradient loss on query tokens using the reward.  
   - Add KL divergence penalty (`β * KL_penalty`, β=0.1), where KL is computed as `KL(P_previous || P_current)` using `torch.nn.functional.kl_div(log P_previous, log P_current)`.  
   - Update LoRA parameters with BitsAndBytes 8-bit Adam optimizer (learning rate=2e-4, gradient norm clipped to 1.0).

6. **Model Management:**  
   - Save the current LoRA state as the "previous model" before each update using a straightforward copy (e.g., `copy.deepcopy(model.lora_params)`).  
   - Use the base model (no adapter) for reward normalization.
   - Save model checkpoints every 100 episodes and on training completion.
   - Maintain the previous model state specifically for KL divergence computation during each policy update.

7. **Logging and Monitoring:**  
   - Log trajectory details and metrics (gradient norms, rewards, loss, KL penalty, filter fraction).  
   - Print periodic console summaries.
   - Use reward as the primary training metric; no additional validation is required.

### 3.2. Mathematical Formulation of the Objective Function

The objective function being optimized can be mathematically expressed as:

$$
\text{Maximize } \mathbb{E}_{\tau \sim \pi, \text{selected}} \left[ \sum_{t=1}^T r_t \right] - \beta \cdot \mathbb{E}_{\tau \sim \pi, \text{selected}} \left[ \sum_{t=1}^T \sum_{i=1}^L \text{KL}(\pi_{\text{previous}}(\cdot | s_{t,i}) || \pi(\cdot | s_{t,i})) \right]
$$

Where:
- $\tau$: A trajectory sampled from the policy $\pi$
- $T$: Number of steps in the trajectory
- $r_t$: Reward at step $t$, defined as the normalized log probability of value tokens
- $s_{t,i} = (context_t, query_t[:i-1])$: State at step $t$, token position $i$
- $L$: Length of each query
- $\pi$: Current policy over query token sequences
- $\pi_{\text{previous}}$: Policy before the update
- $\beta$: KL penalty coefficient (0.1)
- "selected": Trajectories where the average reward exceeds one standard deviation above the historical mean

The corresponding loss function implemented in code is:
$$
\text{Loss} = -\sum_{\text{selected trajectories}} \left[ \sum_{t=1}^T r_t \cdot \sum_{i=1}^L \log \pi(token_{t,i} | s_{t,i}) \right] + \beta \cdot \sum_{\text{selected trajectories}} \left[ \sum_{t=1}^T \sum_{i=1}^L \text{KL}(\pi_{\text{previous}}(\cdot | s_{t,i}) || \pi(\cdot | s_{t,i})) \right]
$$

This implements a standard REINFORCE policy gradient approach with the addition of a KL divergence regularization term.

For numerical stability, the KL divergence between the previous policy and current policy is computed using log probabilities for both distributions:

$$
\text{KL}(\pi_{\text{previous}} || \pi) = \sum_x \exp(\log \pi_{\text{previous}}(x)) \cdot (\log \pi_{\text{previous}}(x) - \log \pi(x))
$$

Which is implemented using PyTorch's `F.kl_div` function with `log_target=True`:

```python
kl_div = F.kl_div(
    F.log_softmax(current_logits, dim=-1),
    F.log_softmax(previous_logits, dim=-1),
    reduction="batchmean",
    log_target=True
)
```

This batched implementation ensures both numerical stability and efficient computation.

### 3.3. Resource Requirements and Error Handling

- **Resource Requirements:**
  - This implementation is designed to run on a single GPU with at least 16GB VRAM.
  - For systems with less memory, reduce batch sizes and/or use gradient accumulation.
  - CPU-only execution is supported but will be significantly slower.
  - Expected training time: 24-48 hours on a modern GPU for a full run.

- **Error Handling Strategy:**
  - Implement assertion statements throughout the codebase to validate shapes, types, and parameter bounds.
  - Add exception handling for common failure modes (e.g., CUDA OOM, file access issues).
  - Checkpointing: Save model state every 100 episodes to recover from interruptions.
  - Do not implement early stopping.
  - Add validation checks for tokenization consistency to prevent query/key/value length mismatches.

---

## 4. Implementation Details

### 4.1. Embeddings Using Llama's Last Layer

- **Extraction:**  
  - Utilize model hooks to capture embeddings from the last attention layer: `(batch_size, sequence_length, embedding_dim)`.  
  - Target module for hooks: `model.model.model.layers[-1].self_attn.q_proj`.
  - Average over `sequence_length`: `(batch_size, embedding_dim)`.
  - Pre-compute and store embeddings during data loading to avoid redundant computation.

- **Grouped Query Attention Handling:**  
  - Dynamically retrieve `num_query_heads`, `num_kv_groups`, and `head_dim` from the model (e.g., via `first_layer.self_attn.k_proj` and `q_proj`).  
  - Reshape query embeddings to `(batch_size, num_query_heads, head_dim)`.  
  - Reshape key embeddings to `(batch_size, num_kv_groups, head_dim)`.  
  - Compute dot products between query and key embeddings per head-group pair, fully leveraging batched operations:
  ```python
  # Reshape for batched computation
  # query_embeddings: [batch_size, embedding_dim] -> [batch_size, num_query_heads, head_dim]
  query_heads = query_embeddings.view(batch_size, num_query_heads, head_dim)
  
  # key_embeddings: [num_keys, batch_size, embedding_dim] -> [num_keys, batch_size, num_kv_groups, head_dim]
  key_groups = key_embeddings.view(num_keys, batch_size, num_kv_groups, head_dim)
  
  # Compute similarities for all batches at once
  similarities = torch.zeros(batch_size, num_query_heads, num_keys, device=query_heads.device)
  
  # Calculate per-head similarities efficiently using batched operations
  for h in range(num_query_heads):
      # Determine which KV group this head should attend to
      kv_group_idx = h // (num_query_heads // num_kv_groups)
      
      # Extract corresponding query heads for all batches [batch_size, head_dim]
      query_head_batch = query_heads[:, h]
      
      # Extract corresponding key groups for all keys and batches [num_keys, batch_size, head_dim]
      key_groups_batch = key_groups[:, :, kv_group_idx]
      
      # Efficiently compute similarity for all keys and all batches at once
      # [batch_size, 1, head_dim] × [num_keys, batch_size, head_dim]ᵀ → [batch_size, num_keys]
      batch_similarities = torch.bmm(
          query_head_batch.unsqueeze(1),
          key_groups_batch.permute(1, 0, 2)
      ).squeeze(1) / math.sqrt(head_dim)
      
      similarities[:, h] = batch_similarities
  
  # Average across heads to get final similarity scores
  # [batch_size, num_heads, num_keys] → [batch_size, num_keys]
  mean_similarities = torch.mean(similarities, dim=1)
  
  # Apply softmax to get probability distribution over keys for each batch
  probabilities = F.softmax(mean_similarities * temperature, dim=-1)
  ```
  - Apply softmax per head to obtain probability distributions over keys for each head.  
  - Average these probability distributions across the head dimension to get a single probability distribution per batch.
  - Efficiently implement using PyTorch's batch operations for maximum performance.

### 4.2. Data Processing and Batching

- **Data Loading:**  
  - Use `AutoTokenizer.from_pretrained(tokenizer_name)` for the tokenizer.  
  - Load Wikipedia articles with `load_dataset("wikipedia", "20220301.en", split="train", streaming=True)`.  
  - Implement functional iterator-based dataloading with methods like `chunk_tokens`, `sample_chunks`, `split_key_values`, and `add_embeddings`.  
  - Filter articles with fewer than `tokens_per_kv_pair * num_kv_pairs` tokens.  
  - Tokenize data using Llama's tokenizer with `add_special_tokens=False` during data loading; store and pass tokenized tensors throughout the pipeline.
  - Streaming approach handles data efficiently without extensive preprocessing or complex edge case handling.

- **Optimization Keys for Data Processing:**
  - **Single-Pass Tokenization**: Tokenize articles once and extract chunks by index operations
  - **Direct Tensor Extraction**: Skip intermediate lists/arrays; convert to tensors early
  - **Batched Operations**: Process articles in configurable batches to maximize throughput
  - **Fixed Dimensions**: Use fixed token lengths to avoid padding and ensure tensor compatibility
  - **Tensor-First Design**: Design data structures around tensors with consistent dimensions

- **Key–Value Pairs:**  
  - Segment articles into fixed-length, non-overlapping pairs (e.g., 20-token keys, 20-token values).  
  - Store as tokenized tensors with fixed dimensions to eliminate repeated tokenization and enable efficient batched operations.
  - Use separators " Query: " and " Response: " with leading and trailing spaces to ensure tokenization consistency.

- **Subset Selection:**  
  - Randomly select a fraction (default: 0.25) of pairs per trajectory.

- **Batching:**  
  - Configurable sizes: `generation_batch_size=64`, `training_batch_size=16`.  
  - Support gradient accumulation.
  - All operations maintain batch dimensions throughout the entire pipeline.

- **Padding-Free Synchronization:**  
  - Enforce fixed token lengths for queries (20 tokens), keys (20 tokens), and values (20 tokens) to maintain synchronized batch dimensions without padding.  
  - Verify tokenization consistency with tests.
  - Pre-tokenize all text data to avoid repeated tokenization in the inner loops.

### 4.3. Prompting the Base Model

- **Unified Prompt Format:**  
  - Use " Query: _ Response: _" with leading and trailing spaces.  
  - Start each trajectory with the first key-value pair from the article included in the context.

- **For Query Generation:**  
  - Prompt ends with " Query:" to signal the model to generate the next query.  
  - Generate a fixed number of tokens (e.g., 20 tokens).

- **For Value Prediction and Training:**  
  - Use the full sequence, including " Query: [queryN+1] Response: [responseN+1]".  
  - Compute log probabilities for tokens following " Response:" in the final pair.  
  - *Clarification:* The context for reward computation includes the " Response: " prefix.

### 4.4. Functional Programming and Codebase Structure

- **Pure Functions and Iterators:**  
  - Implement dataloading with iterators (e.g., `iter_wikipedia_articles()`) to yield processed articles in a functional style.  
  - Define pure functions for key operations: `extract_embeddings`, `compute_similarity`, `sample_key_value`, `calculate_reward`.

- **Batched Tensor-Based Data Structures:**  
  - Use dataclasses optimized for batched operations with tokenized tensors as the primary storage mechanism:
    ```python
    @dataclass
    class KeyValuePair:
        """Dataclass for a key-value pair optimized for batched processing."""
        key_tokens: torch.Tensor  # Shape: [batch_size, TOKENS_PER_KEY]
        value_tokens: torch.Tensor  # Shape: [batch_size, TOKENS_PER_VALUE]
        key_embedding: torch.Tensor  # Shape: [batch_size, embedding_dim]
        key_text: List[str]  # For logging and debugging
        value_text: List[str]  # For logging and debugging
        
        # Implement tensor shape validation here
        
    @dataclass
    class TrajectoryStep:
        context_tokens: torch.Tensor  # Shape: [batch_size, context_length]
        query_tokens: torch.Tensor  # Shape: [batch_size, TOKENS_PER_KEY]
        selected_pair: KeyValuePair
        reward: torch.Tensor  # Shape: [batch_size]
        similarity_score: torch.Tensor  # Shape: [batch_size]
        context_text: List[str]  # For logging and debugging
        query_text: List[str]  # For logging and debugging
        
    @dataclass
    class Trajectory:
        steps: List[TrajectoryStep]
        avg_reward: torch.Tensor  # Shape: [batch_size]
        total_reward: torch.Tensor  # Shape: [batch_size]
    ```
  
  - All data structures maintain fixed tensor dimensions to avoid padding and enable efficient batched operations.
  - Original text is preserved only for logging and debugging purposes; all calculations operate on pre-tokenized tensors.

- **Implementation Hints for Key Data Iterator:**
  
  When implementing the data iterator, keep these points in mind:
  
  ```python
  # Key pseudocode for the data iterator
  def iter_key_value_pairs(batch_size, embedding_fn):
      # 1. Get tokenizer
      # 2. Process articles in batches of size batch_size
      # 3. For each article batch:
      #    a. Tokenize each article (once only)
      #    b. Extract key-value pairs using direct index arithmetic:
      #       - key_tokens = tokens[i:i+TOKENS_PER_KEY]
      #       - value_tokens = tokens[i+TOKENS_PER_KEY:i+TOKENS_PER_KEY+TOKENS_PER_VALUE]
      #    c. Convert to tensors and compute embeddings
      #    d. Create and yield KeyValuePair object
  ```
  
  This approach has several advantages:
  - Tokenization happens once per article, not repeatedly for each chunk
  - Direct index-based extraction is much faster than iterative chunking
  - Batch operations maintain tensor dimensions for consistent processing

- **Avoiding Common Implementation Pitfalls:**
  - Add the `truncation=True` parameter to tokenization calls to handle longer inputs
  - Use tensor views instead of copies when possible for efficient memory usage
  - Validate tensor shapes early to catch dimension mismatches
  - When tokenizing batches, use `return_tensors="pt"` for direct tensor outputs
  - Calculate the minimum required tokens as `(TOKENS_PER_KEY + TOKENS_PER_VALUE) * NUM_KV_PAIRS`

- **Batched Operations:**  
  - All functions are designed to work with batch dimensions for maximum efficiency.
  - Tokenization happens once during data loading, with all subsequent operations working directly on tokenized tensors.
  - Similarity calculations, reward computations, and policy updates all operate on batches to minimize processing overhead.

### 4.5. Trajectory Generation and Training Loop

- **Trajectory Generation:**  
  - Process multiple trajectories in parallel using batched operations.
  - Start each trajectory with the first key-value pair from its article as an in-context example.
  - For each step:
    - Generate queries in batch using the current model.
    - Extract query embeddings in batch from the last attention layer.
    - Compute similarities with remaining key embeddings using batched grouped query attention.
    - Sample keys based on similarity, remove them from available pools.
    - Add selected key-value pairs to contexts.
    - Compute reward normalization on-the-fly using counterfactual queries from the pre-trained model, all in batch.

- **Training:**  
  - Filter high-reward trajectories using a moving average of reward statistics (mean and standard deviation computed over the entire trajectory history).
  - Compute policy gradient loss on query tokens using the batched REINFORCE algorithm:
    ```python
    # Process multiple trajectories simultaneously
    loss = 0
    batch_size = len(selected_trajectories)
    
    # Collect all queries and rewards from selected trajectories
    all_queries = []  # Will contain tensors of shape [seq_len]
    all_rewards = []  # Will contain scalar values
    
    for trajectory in selected_trajectories:
        all_queries.extend([step.query_tokens for step in trajectory.steps])
        # Expand rewards to match queries (same reward for each query in a trajectory)
        step_rewards = [trajectory.avg_reward] * len(trajectory.steps)
        all_rewards.extend(step_rewards)
    
    # Batch process all queries (much more efficient than one at a time)
    # Create a single batch of all queries from all trajectories
    query_batch = torch.stack(all_queries)  # [total_queries, seq_len]
    reward_batch = torch.tensor(all_rewards, device=query_batch.device)  # [total_queries]
    
    # Compute log probabilities for the entire batch at once
    with torch.no_grad():
        query_log_probs = compute_batch_token_log_probs(query_batch, model)  # [total_queries, seq_len]
    
    # Apply policy gradient with vector reward
    policy_loss = -(query_log_probs.sum(dim=1) * reward_batch).sum()
    
    # Compute KL divergence penalty for the batch
    kl_loss = compute_kl_divergence(previous_model, current_model, tokenizer, query_batch)
    
    # Combine losses
    loss = policy_loss + beta * kl_loss
    ```
  - Add KL divergence penalty (`β=0.1`) and update LoRA parameters using batched operations.
  - Run for a specified number of episodes (`num_episodes`), using reward as the primary metric.

The key advantage of this batched approach is that it minimizes sequential processing, allowing the model to handle multiple trajectories simultaneously during both generation and training phases. This significantly improves computational efficiency on modern hardware.

### 4.6. Model Management with LoRA

- Use a single LoRA adapter with standard parameters (rank=8, alpha=16, dropout=0.05); save its state as the "previous model" before each update using a straightforward copy (e.g., `copy.deepcopy(model.lora_params)`).  
- Use the base model (no adapter) for reward normalization.

### 4.7. Testing and Incremental Development

- **Pytest Tests:**  
  - Validate data processing (iterator output, token filtering), query generation, reward calculation, trajectory filtering (moving average), and training loop.  
  - Include tests for tokenization consistency (e.g., spaces around " Query: " and " Response: ", exact token counts).  
  - *Clarification:* Extensive test cases are needed to address potential tokenization failure modes.

- **Development Approach:**  
  - Build and test modules incrementally (e.g., dataloader, embeddings, RL agent), ensuring each component functions independently before integration.
  - Begin with minimal functionality and extend only after validation.
  - Implement the pure data processing pipeline first, then add embedding extraction.

- **Error Recovery and Diagnostics:**
  - Implement robust error logging that includes hardware state (CUDA memory usage, etc.).
  - Create diagnostic tools to visualize query-key similarity distributions.
  - Add memory profiling during development to optimize batch sizes.
  - Design environment validation in the setup phase to verify all requirements are met.
  - Implement graceful degradation for systems with limited resources.

---

## 5. Repository Structure

```
attention-guided-rl/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py              # Entry point for training
│   ├── model.py             # Llama-3.2-3B and LoRA setup
│   ├── embeddings.py        # Embedding extraction and similarity computation
│   ├── data.py              # Functional iterator-based dataloader
│   ├── utils.py             # Helper functions (e.g., logging, reward computation)
│   └── config.py            # Configuration parameters
└── tests/
    ├── test_model.py        # Model initialization and inference tests
    ├── test_embeddings.py   # Embedding and similarity tests
    ├── test_data.py         # Dataloader and token filtering tests
    └── test_training.py     # Training loop and RL agent tests
```

---

## 6. Conclusion

This design document provides a comprehensive blueprint for implementing an RL-driven, self-directed language model training system using Wikipedia data. By integrating Llama-3.2-3B with LoRA, leveraging attention-guided embeddings with grouped query attention, ensuring tokenization consistency with spaces around prompts, filtering articles based on required token counts, and adopting a functional dataloading approach with iterators, the system achieves efficient and stable learning. Extensive testing and modularity ensure reliability and reproducibility. This standalone guide equips developers with a clear, detailed plan to construct a PyTorch-based software repository.