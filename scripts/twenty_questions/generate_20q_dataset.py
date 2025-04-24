import os
import json
import time
from anthropic import Anthropic

# Initialize Anthropic client
# You need to set your API key in the environment
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

client = Anthropic(api_key=API_KEY)

# Use the latest Claude Sonnet model
MODEL = "claude-3-7-sonnet-20250219"

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_questions(num_questions=20):
    """Generate a list of strong boolean questions for a game of 20 questions."""
    prompt = f"""
    Please generate {num_questions} strong, boolean-valued (yes/no) questions that would be effective
    in a game of 20 questions. These questions should help narrow down what an object is.
    
    The questions should be general enough to apply to any object, and should be designed to
    efficiently split the space of possible objects.
    
    Format your response as a JSON array of strings, with each string being a question.
    Example: ["Is it larger than a breadbox?", "Is it a living thing?", ...]
    """
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0.2,
        system="You are a helpful assistant that generates questions for a 20 questions game. Reply only with a valid JSON array of questions.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract JSON array from the response
    content = response.content[0].text
    # Find JSON array in the response (it might have additional text)
    start_idx = content.find('[')
    end_idx = content.rfind(']') + 1
    json_str = content[start_idx:end_idx]
    
    questions = json.loads(json_str)
    return questions

def generate_objects(num_objects=100):
    """Generate a diverse list of objects for the 20 questions game."""
    prompt = f"""
    Please generate {num_objects} diverse objects that could be used in a game of 20 questions.
    Include a variety of items: animals, plants, household objects, tools, vehicles, foods, etc.
    
    Format your response as a JSON array of strings, with each string being an object name.
    Example: ["elephant", "screwdriver", "pizza", "airplane", ...]
    """
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0.7,  # Higher temperature for more diversity
        system="You are a helpful assistant that generates a diverse list of objects. Reply only with a valid JSON array of object names.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract JSON array from the response
    content = response.content[0].text
    # Find JSON array in the response
    start_idx = content.find('[')
    end_idx = content.rfind(']') + 1
    json_str = content[start_idx:end_idx]
    
    objects = json.loads(json_str)
    return objects

def get_answers_for_object(object_name, questions):
    """Get yes/no answers for an object given a list of questions."""
    prompt = f"""
    Object: {object_name}
    
    For the following {len(questions)} questions, please answer YES or NO as they apply to the object.
    For each question, think carefully about the properties of the object and provide an accurate answer.
    
    Questions:
    {json.dumps(questions, indent=2)}
    
    Format your response as a JSON array of strings, where each string is either "YES" or "NO",
    corresponding to the answer for each question in order.
    Example: ["YES", "NO", "YES", ...]
    """
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0.1,  # Low temperature for consistency
        system="You are a helpful assistant that provides accurate yes/no answers to questions about objects. Reply only with a valid JSON array of YES/NO answers.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract JSON array from the response
    content = response.content[0].text
    # Find JSON array in the response
    start_idx = content.find('[')
    end_idx = content.rfind(']') + 1
    json_str = content[start_idx:end_idx]
    
    answers = json.loads(json_str)
    return answers

def load_existing_dataset(output_file):
    """Load an existing dataset file if it exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing file {output_file}, starting fresh.")
    return None

def generate_objects_in_batches(total_needed, batch_size=50):
    """Generate objects in smaller batches to avoid API limitations."""
    all_objects = []
    remaining = total_needed
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        print(f"Generating batch of {current_batch} objects...")
        
        try:
            batch_objects = generate_objects(current_batch)
            all_objects.extend(batch_objects)
            remaining -= current_batch
            print(f"Successfully generated {len(batch_objects)} objects. {remaining} remaining.")
            
            # Add a small delay between batches
            if remaining > 0:
                time.sleep(1.0)
                
        except Exception as e:
            print(f"Error generating batch: {str(e)}")
            print(f"Retrying with smaller batch size...")
            # If we fail, try with half the batch size, but minimum 5
            batch_size = max(5, batch_size // 2)
    
    return all_objects

def create_dataset(num_questions=20, num_objects=100, output_file=os.path.join(DATA_DIR, "20q_dataset.json"), resume=False):
    """Create a dataset for the 20 questions game."""
    
    # Check for existing dataset
    existing_dataset = None
    if resume:
        existing_dataset = load_existing_dataset(output_file)
    
    if existing_dataset:
        print(f"Resuming from existing dataset in {output_file}")
        dataset = existing_dataset
        questions = dataset["questions"]
        
        # Find which objects are already processed
        processed_objects = set(item["object"] for item in dataset["data"])
        num_already_processed = len(processed_objects)
        
        # If we have the full object list in the dataset
        if "all_objects" in dataset:
            existing_objects = dataset["all_objects"]
            
            # Check if we need to generate more objects
            if num_objects > len(existing_objects):
                print(f"Existing dataset has {len(existing_objects)} objects, but {num_objects} requested.")
                additional_needed = num_objects - len(existing_objects)
                print(f"Generating {additional_needed} additional objects in batches...")
                
                # Generate objects in smaller batches
                additional_objects = generate_objects_in_batches(additional_needed)
                
                # Add new objects to the list, avoiding duplicates
                existing_object_set = set(existing_objects)
                unique_new_objects = [obj for obj in additional_objects if obj not in existing_object_set]
                
                # Update the full object list
                objects = existing_objects + unique_new_objects
                print(f"Added {len(unique_new_objects)} new unique objects. Total objects: {len(objects)}")
                dataset["all_objects"] = objects
                
                # Save immediately after adding objects in case of later errors
                with open(output_file, "w") as f:
                    json.dump(dataset, f, indent=2)
            else:
                objects = existing_objects
                print(f"Found {len(objects)} total objects, {len(processed_objects)} already processed.")
            
            # Filter out objects that have already been processed
            objects_to_process = [obj for obj in objects if obj not in processed_objects]
            print(f"Continuing with remaining {len(objects_to_process)} objects.")
        else:
            # Generate new objects if we don't have the full list
            print(f"Generating {num_objects} new objects in batches...")
            objects = generate_objects_in_batches(num_objects)
            # Store the full object list for future resuming
            dataset["all_objects"] = objects
            objects_to_process = [obj for obj in objects if obj not in processed_objects]
            print(f"Generated {len(objects)} objects, {len(processed_objects)} already processed.")
            print(f"Continuing with remaining {len(objects_to_process)} objects.")
            
            # Save immediately after generating objects
            with open(output_file, "w") as f:
                json.dump(dataset, f, indent=2)
    else:
        # Start fresh
        print(f"Generating {num_questions} questions...")
        questions = generate_questions(num_questions)
        print("Questions generated.")
        
        print(f"Generating {num_objects} new objects in batches...")
        objects = generate_objects_in_batches(num_objects)
        print("Objects generated.")
        
        dataset = {
            "questions": questions,
            "all_objects": objects,  # Store all objects for resuming
            "data": []
        }
        
        processed_objects = set()
        num_already_processed = 0
        objects_to_process = objects
        
        # Save immediately after initial generation
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)
    
    # Process objects
    print("Getting answers for each object...")
    for i, obj in enumerate(objects_to_process):
        # Skip if already processed
        if obj in processed_objects:
            continue
            
        current_processed = num_already_processed + i
        print(f"Processing object {i+1}/{len(objects_to_process)} ({current_processed+1}/{len(objects)}): {obj}")
        
        try:
            answers = get_answers_for_object(obj, questions)
            
            # Add to dataset
            dataset["data"].append({
                "object": obj,
                "answers": answers
            })
            
            # Update processed objects set
            processed_objects.add(obj)
            
            # Save progress after each object in case of interruption
            with open(output_file, "w") as f:
                json.dump(dataset, f, indent=2)
            
            # Respect rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing object '{obj}': {str(e)}")
            print("Saving progress and continuing with next object...")
            # Save progress after each object in case of interruption
            with open(output_file, "w") as f:
                json.dump(dataset, f, indent=2)
    
    print(f"Dataset created and saved to {output_file}")
    print(f"Total objects: {len(objects)}, Processed objects: {len(processed_objects)}")
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a dataset for 20 questions game")
    parser.add_argument("--questions", type=int, default=20, help="Number of questions to generate")
    parser.add_argument("--objects", type=int, default=100, help="Number of objects to generate")
    parser.add_argument("--output", type=str, default=os.path.join(DATA_DIR, "20q_dataset.json"), help="Output file path")
    parser.add_argument("--resume", action="store_true", help="Resume from existing dataset")
    
    args = parser.parse_args()
    
    create_dataset(args.questions, args.objects, args.output, args.resume) 