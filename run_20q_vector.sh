#!/bin/bash
# Script to run a larger training session with Twenty Questions dataset and vector queries

echo "Starting Twenty Questions training with vector queries..."
echo "Configuration:"
echo "- Dataset: Twenty Questions"
echo "- Query type: Vector queries"
echo "- Episodes: 1000"
echo "- Batch size: 4"
echo "- Learning rate: 2e-4"
echo "- Training percentile: 90.0 (top 10%)"
echo ""

# Create a timestamped run name
RUN_NAME="20q_vector_$(date +%Y%m%d_%H%M%S)"

# Run the training
python -m src.main \
    --dataset twenty_questions \
    --use-vector-queries \
    --episodes 1000 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --training-percentile 90.0 \
    --log-interval 10 \
    --run-name "$RUN_NAME" \
    --verbose

echo ""
echo "Training complete! Run name: $RUN_NAME"
echo "Check logs/ directory for results and plots" 