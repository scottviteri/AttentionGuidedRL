# Twenty Questions Dataset Generator

This script generates a dataset for a game of 20 questions using the Claude API. It creates:

1. A set of strong boolean-valued (yes/no) questions for an unspecified game of 20 questions
2. A diverse set of random objects
3. Yes/No answers from Claude for each object against all 20 questions

Each datapoint in the resulting dataset consists of an object and its 20 corresponding yes/no answers.

## Requirements

- Python 3.7+
- Anthropic Python client (`pip install anthropic`)
- Claude API key (obtained from [Anthropic's platform](https://console.anthropic.com/))
- For analysis: matplotlib, numpy

## Setup

1. Install the required packages:
   ```bash
   pip install anthropic matplotlib numpy
   ```

2. Set your API key as an environment variable:
   ```bash
   export ANTHROPIC_API_KEY="your_api_key_here"
   ```

## Dataset Generation

Run the script with default parameters:
```bash
python scripts/twenty_questions/generate_20q_dataset.py
```

### Command-line Arguments

- `--questions`: Number of questions to generate (default: 20)
- `--objects`: Number of objects to generate (default: 100)
- `--output`: Output file path (default: "data/20q_dataset.json")
- `--resume`: Resume from an existing dataset file (default: False)

Example with custom parameters:
```bash
python scripts/twenty_questions/generate_20q_dataset.py --questions 20 --objects 50 --output data/my_dataset.json
```

### Resuming Interrupted Runs

If the script is interrupted for any reason (API errors, network issues, etc.), you can resume from where it left off:

```bash
python scripts/twenty_questions/generate_20q_dataset.py --output data/my_dataset.json --resume
```

The script will:
1. Load the existing dataset file
2. Skip objects that have already been processed
3. Continue with the remaining objects in the list

## Dataset Analysis

The repo includes an analysis script that provides insights into the generated dataset:

```bash
python scripts/twenty_questions/analyze_20q_dataset.py --input data/20q_dataset.json --visualize
```

The analysis provides:
- Basic statistics (number of questions and objects)
- Question bias (percentage of YES answers for each question)
- Most discriminating questions (based on entropy)
- Similar object pairs (based on answer patterns)
- Visualizations of question bias and entropy (when `--visualize` is used)

### Command-line Arguments

- `--input`: Input dataset file path (default: "data/20q_dataset.json")
- `--visualize`: Generate visualizations (saves PNG files to the "visualizations/" directory)

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "questions": [
    "Is it larger than a breadbox?",
    "Is it a living thing?",
    ...
  ],
  "all_objects": [
    "elephant",
    "screwdriver",
    ...
  ],
  "data": [
    {
      "object": "elephant",
      "answers": ["YES", "YES", ...]
    },
    {
      "object": "screwdriver",
      "answers": ["NO", "NO", ...]
    },
    ...
  ]
}
```

## Notes

- The script uses Claude 3.7 Sonnet, the latest Sonnet model from Anthropic.
- The script saves progress after processing each object, so if it's interrupted, you can resume with the `--resume` flag.
- Error handling is implemented to continue processing even if individual API calls fail.
- Rate limiting is implemented with a small delay between API calls.

## Integration with AttentionGuidedRL

This dataset is intended to be used for training and evaluating the attention-guided reinforcement learning system. The generated data can be loaded using the data loaders in `src/data.py` and used for:

1. Training the base model to predict answers based on objects
2. Testing the attention mechanism's ability to select relevant question-answer pairs
3. Evaluating the overall system's performance in building coherent trajectories 