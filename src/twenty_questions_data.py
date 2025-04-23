"""
Twenty Questions dataset loader for the Attention-Guided RL project.

Contains functions for loading and processing the twenty questions dataset
for use with the reinforcement learning training loop.
"""

import os
import json
import torch
import random
from typing import Dict, Iterator, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

from transformers import PreTrainedTokenizer

from src.config import (
    DEVICE,
    TOKENS_PER_KEY,
    TOKENS_PER_VALUE,
    TOKENS_PER_QUERY,
    QUERY_PREFIX,
    KEY_PREFIX,
    VALUE_PREFIX,
)

from src.data import QKVStep


def load_twenty_questions_dataset(dataset_path: str = None) -> Dict:
    """
    Load the twenty questions dataset from a JSON file.

    Args:
        dataset_path: Path to the dataset JSON file

    Returns:
        Dict: The dataset as a dictionary
    """
    if dataset_path is None:
        # Use default path
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "20q_dataset.json"
        )

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    return dataset


def encode_question_answer_pair(
    question: str,
    answer: str,
    tokenizer: PreTrainedTokenizer,
    pad_to_key_length: bool = True,
    pad_to_value_length: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """
    Encode a question-answer pair as key-value tokens.

    Args:
        question: The question text
        answer: The answer text (YES/NO)
        tokenizer: The tokenizer to use
        pad_to_key_length: Whether to pad the key to TOKENS_PER_KEY
        pad_to_value_length: Whether to pad the value to TOKENS_PER_VALUE

    Returns:
        Tuple: (key_tokens, value_tokens, key_text, value_text)
    """
    # Encode the question as the key
    key_ids = tokenizer.encode(
        question, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=TOKENS_PER_KEY
    )
    
    # Encode the answer as the value
    value_ids = tokenizer.encode(
        answer, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=TOKENS_PER_VALUE
    )
    
    # Pad or truncate to the required lengths
    if pad_to_key_length:
        key_tokens = torch.zeros(TOKENS_PER_KEY, dtype=torch.long, device=DEVICE)
        key_tokens[:len(key_ids)] = torch.tensor(key_ids, dtype=torch.long, device=DEVICE)
    else:
        key_tokens = torch.tensor(key_ids, dtype=torch.long, device=DEVICE)
    
    if pad_to_value_length:
        value_tokens = torch.zeros(TOKENS_PER_VALUE, dtype=torch.long, device=DEVICE)
        value_tokens[:len(value_ids)] = torch.tensor(value_ids, dtype=torch.long, device=DEVICE)
    else:
        value_tokens = torch.tensor(value_ids, dtype=torch.long, device=DEVICE)
    
    return key_tokens, value_tokens, question, answer


def generate_object_trajectory(
    object_data: Dict,
    questions: List[str],
    tokenizer: PreTrainedTokenizer,
    embedding_fn: Callable,
    num_questions: int = 5
) -> List[QKVStep]:
    """
    Generate a trajectory of question-answer pairs for a single object.

    Args:
        object_data: The object data containing the object name and answers
        questions: List of all available questions
        tokenizer: The tokenizer to use
        embedding_fn: Function to compute embeddings for keys
        num_questions: Number of question-answer pairs to include in the trajectory

    Returns:
        List[QKVStep]: A list of QKVStep objects forming a trajectory
    """
    object_name = object_data["object"]
    answers = object_data["answers"]
    
    # Select a subset of question indices to use
    question_indices = random.sample(range(len(questions)), min(num_questions, len(questions)))
    
    # Sort indices to maintain the original order
    question_indices.sort()
    
    # Create a trajectory of question-answer pairs
    trajectory = []
    
    for idx in question_indices:
        question = questions[idx]
        answer = answers[idx]
        
        # Encode the question-answer pair
        key_tokens, value_tokens, key_text, value_text = encode_question_answer_pair(
            question, answer, tokenizer
        )
        
        # Expand to batch size 1
        key_tokens = key_tokens.unsqueeze(0)
        value_tokens = value_tokens.unsqueeze(0)
        
        # Compute key embedding
        key_embedding = embedding_fn(key_tokens)
        
        # Create a QKVStep
        qkv_step = QKVStep(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=[key_text],
            value_text=[value_text]
        )
        
        trajectory.append(qkv_step)
    
    return trajectory


def create_twenty_questions_context(
    tokenizer: PreTrainedTokenizer
) -> torch.Tensor:
    """
    Create an initial context for the 20 questions game without revealing the object.

    Args:
        tokenizer: The tokenizer to use

    Returns:
        torch.Tensor: The encoded context [1, seq_length]
    """
    # Create a prompt for playing 20 questions with detailed instructions
    prompt = (
        "I am thinking of an object. You are playing 20 questions to guess what it is. "
        "You need to ask yes/no questions that will help you identify the object efficiently. "
        "Try to ask questions that will divide the space of possible objects in half with each question."
    )
    
    # Encode the prompt
    context_ids = tokenizer.encode(
        prompt, 
        add_special_tokens=False
    )
    
    # Convert to tensor and add batch dimension
    context_tokens = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)
    
    return context_tokens


def iter_twenty_questions_batches(
    batch_size: int = 1,
    embedding_fn: Callable = None,
    tokenizer: PreTrainedTokenizer = None,
    dataset_path: str = None,
    num_questions_per_trajectory: int = 5,
    shuffle: bool = True
) -> Iterator[Tuple[List[QKVStep], torch.Tensor]]:
    """
    Create an iterator that yields batches of trajectories from the twenty questions dataset.

    Args:
        batch_size: Number of objects to process in each batch
        embedding_fn: Function to compute embeddings for keys
        tokenizer: The tokenizer to use
        dataset_path: Path to the dataset JSON file
        num_questions_per_trajectory: Number of question-answer pairs per object
        shuffle: Whether to shuffle the objects

    Returns:
        Iterator[Tuple[List[QKVStep], torch.Tensor]]: Iterator yielding (trajectory, context_tokens)
    """
    # Load the dataset
    dataset = load_twenty_questions_dataset(dataset_path)
    questions = dataset["questions"]
    data = dataset["data"]
    
    # Create indices for all objects
    indices = list(range(len(data)))
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(indices)
    
    # Yield batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = [data[idx] for idx in batch_indices]
        
        # Process each object in the batch
        batch_trajectories = []
        batch_contexts = []
        
        for obj_data in batch_data:
            # Generate trajectory for this object
            trajectory = generate_object_trajectory(
                obj_data,
                questions,
                tokenizer,
                embedding_fn,
                num_questions_per_trajectory
            )
            
            # Create a generic 20 questions context (without revealing the object)
            context_tokens = create_twenty_questions_context(tokenizer)
            
            batch_trajectories.append(trajectory)
            batch_contexts.append(context_tokens)
        
        # Yield each object's trajectory and context separately
        for traj, ctx in zip(batch_trajectories, batch_contexts):
            yield traj, ctx


def get_twenty_questions_pool(
    object_data: Dict,
    questions: List[str],
    tokenizer: PreTrainedTokenizer,
    embedding_fn: Callable,
) -> List[QKVStep]:
    """
    Get a pool of all question-answer pairs for a given object.

    Args:
        object_data: The object data containing the object name and answers
        questions: List of all available questions
        tokenizer: The tokenizer to use
        embedding_fn: Function to compute embeddings for keys

    Returns:
        List[QKVStep]: A list of QKVStep objects for all questions
    """
    object_name = object_data["object"]
    answers = object_data["answers"]
    
    # Create a QKVStep for each question-answer pair
    pool = []
    
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        # Encode the question-answer pair
        key_tokens, value_tokens, key_text, value_text = encode_question_answer_pair(
            question, answer, tokenizer
        )
        
        # Expand to batch size 1
        key_tokens = key_tokens.unsqueeze(0)
        value_tokens = value_tokens.unsqueeze(0)
        
        # Compute key embedding
        key_embedding = embedding_fn(key_tokens)
        
        # Create a QKVStep
        qkv_step = QKVStep(
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            key_embedding=key_embedding,
            key_text=[key_text],
            value_text=[value_text]
        )
        
        pool.append(qkv_step)
    
    return pool 