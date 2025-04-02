import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
# Create output directory for visualizations
VIZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

def load_dataset(file_path):
    """Load the 20 questions dataset."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_dataset(dataset):
    """Analyze the dataset and print statistics."""
    questions = dataset['questions']
    data = dataset['data']
    
    # Basic stats
    num_questions = len(questions)
    num_objects = len(data)
    
    print(f"Dataset Statistics:")
    print(f"Number of questions: {num_questions}")
    print(f"Number of objects: {num_objects}")
    
    # Analyze question bias
    yes_counts = [0] * num_questions
    
    for item in data:
        answers = item['answers']
        for i, answer in enumerate(answers):
            if answer.upper() == "YES":
                yes_counts[i] += 1
    
    yes_percentages = [count / num_objects * 100 for count in yes_counts]
    
    print("\nQuestion Bias (% of YES answers):")
    for i, (question, percentage) in enumerate(zip(questions, yes_percentages)):
        print(f"Q{i+1:2d}: {percentage:5.1f}% YES - {question}")
    
    # Identify most discriminating questions
    entropies = []
    for i, count in enumerate(yes_counts):
        # Calculate entropy: -p(yes)log(p(yes)) - p(no)log(p(no))
        p_yes = count / num_objects
        p_no = 1 - p_yes
        
        if p_yes == 0 or p_yes == 1:  # Avoid log(0)
            entropy = 0
        else:
            entropy = -p_yes * np.log2(p_yes) - p_no * np.log2(p_no)
        entropies.append(entropy)
    
    # Sort questions by entropy (most discriminating first)
    sorted_questions = sorted(zip(range(num_questions), questions, entropies), 
                             key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 Most Discriminating Questions (Highest Entropy):")
    for i, (idx, question, entropy) in enumerate(sorted_questions[:5]):
        print(f"{i+1}. Q{idx+1}: {question} (Entropy: {entropy:.3f})")
    
    # Check for most similar objects (based on answer patterns)
    print("\nFinding similar objects based on answer patterns...")
    similarity_pairs = []
    
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            obj1 = data[i]
            obj2 = data[j]
            
            # Count matching answers
            matches = sum(1 for a1, a2 in zip(obj1['answers'], obj2['answers']) if a1 == a2)
            similarity = matches / num_questions
            
            similarity_pairs.append((obj1['object'], obj2['object'], similarity))
    
    # Sort by similarity (most similar first)
    similarity_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 Most Similar Object Pairs:")
    for i, (obj1, obj2, similarity) in enumerate(similarity_pairs[:5]):
        print(f"{i+1}. {obj1} & {obj2}: {similarity*100:.1f}% match ({int(similarity*num_questions)}/{num_questions} questions)")
    
    return {
        'yes_percentages': yes_percentages,
        'entropies': entropies,
        'sorted_questions': sorted_questions,
        'similarity_pairs': similarity_pairs
    }

def visualize_dataset(dataset, analysis_results):
    """Visualize the dataset analysis results."""
    yes_percentages = analysis_results['yes_percentages']
    entropies = analysis_results['entropies']
    
    # 1. Question bias visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(yes_percentages) + 1), yes_percentages)
    plt.axhline(y=50, color='r', linestyle='--')
    plt.title('Question Bias: Percentage of YES Answers')
    plt.xlabel('Question Number')
    plt.ylabel('% YES')
    plt.xticks(range(1, len(yes_percentages) + 1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    viz_path = os.path.join(VIZ_DIR, 'question_bias.png')
    plt.savefig(viz_path)
    print(f"Saved question bias visualization to '{viz_path}'")
    
    # 2. Question discrimination power (entropy)
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(entropies) + 1), entropies)
    plt.title('Question Discrimination Power (Entropy)')
    plt.xlabel('Question Number')
    plt.ylabel('Entropy')
    plt.xticks(range(1, len(entropies) + 1))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    viz_path = os.path.join(VIZ_DIR, 'question_entropy.png')
    plt.savefig(viz_path)
    print(f"Saved question entropy visualization to '{viz_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a 20 Questions dataset")
    parser.add_argument("--input", type=str, default=os.path.join(DATA_DIR, "20q_dataset.json"), help="Input dataset file path")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.input)
    
    # Analyze dataset
    analysis_results = analyze_dataset(dataset)
    
    # Visualize if requested
    if args.visualize:
        visualize_dataset(dataset, analysis_results) 