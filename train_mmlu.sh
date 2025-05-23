#!/bin/bash

# Default parameters
MODEL_TYPE="mistral"
BATCH_SIZE=8
EI_VALUE=1.0
COT_LENGTH=150
TEMPERATURE=0.8
NUM_BATCHES=10000
SUBJECT=""
SPLIT="validation"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --ei)
      EI_VALUE="$2"
      shift 2
      ;;
    --cot_length)
      COT_LENGTH="$2"
      shift 2
      ;;
    --temp)
      TEMPERATURE="$2"
      shift 2
      ;;
    --batches)
      NUM_BATCHES="$2"
      shift 2
      ;;
    --subject)
      SUBJECT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model MODEL_TYPE      Model type (default: mistral)"
      echo "  --batch_size SIZE       Batch size (default: 8)"
      echo "  --ei VALUE              Expert Iteration value (default: 1.0)"
      echo "  --cot_length LENGTH     Chain of thought length (default: 150)"
      echo "  --temp TEMPERATURE      Sampling temperature (default: 0.8)"
      echo "  --batches NUM           Number of batches (default: 10000)"
      echo "  --subject SUBJECT       MMLU subject to train on (default: all subjects)"
      echo "  --split SPLIT           MMLU dataset split (default: validation)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the subject argument if specified
SUBJECT_ARG=""
if [ ! -z "$SUBJECT" ]; then
  SUBJECT_ARG="--mmlu_subject $SUBJECT"
fi

# Run the training
python src/train.py \
  --task_type mmlu \
  --model_type $MODEL_TYPE \
  --batch_size $BATCH_SIZE \
  --use_ei $EI_VALUE \
  --cot_length $COT_LENGTH \
  --temperature $TEMPERATURE \
  --num_batches $NUM_BATCHES \
  --mmlu_split $SPLIT \
  $SUBJECT_ARG

echo "Training completed!" 