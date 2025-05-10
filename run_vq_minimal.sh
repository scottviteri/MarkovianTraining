#!/bin/bash
# Run the minimal Vector Quantization training implementation

# Default parameters
MODEL="gpt2"
LEARNING_RATE=1e-4
BATCH_SIZE=32
NUM_BATCHES=10000
VQ_REASON_LENGTH=25
VQ_SAMPLING_TEMP=0 # Sampling temperature (higher = more diversity)
PRINT_FREQUENCY=20
LORA_RANK=0
TASK_TYPE="wiki_continuation"
CONTEXT_LENGTH=25
MAX_TARGET_LENGTH=25
#FILLER_TOKEN="<REASONING>"
DEBUG_REPEAT=true # Changed default for clarity
DEBUG_GRADIENTS=false
CODEBOOK_LOSS_WEIGHT=0 # New default parameter for codebook loss weight
COPY_TEST=0  # Run copy test every N batches (0 to disable)
COPY_TEST_SAMPLES=1  # Number of samples for copy test
VQ_USE_ARGMAX=true # New default parameter for using argmax instead of sampling
ACTOR_HIDDEN_LAYER_INDEX=-1 
NORMALIZE_REASONING_STATES=true # New: Default to normalizing actor reasoning states
VQ_SEQUENTIAL_GENERATION=false # New default: sequential generation for VQ
CHECKPOINT_FREQUENCY=0 # Default checkpoint frequency
PLOT_FREQUENCY=30 # New: Default plot frequency
DEBUG_ANSWER_IS_QUESTION=true # New: Default for answer is question debug
USE_GUMBEL_SOFTMAX_VQ=true # New: Use Gumbel-Softmax VQ
GUMBEL_TAU=1.0              # New: Tau for Gumbel-Softmax

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_batches)
      NUM_BATCHES="$2"
      shift 2
      ;;
    --vq_reason_length)
      VQ_REASON_LENGTH="$2"
      shift 2
      ;;
    --vq_sampling_temp)
      VQ_SAMPLING_TEMP="$2"
      shift 2
      ;;
    --print_frequency)
      PRINT_FREQUENCY="$2"
      shift 2
      ;;
    --lora_rank)
      LORA_RANK="$2"
      shift 2
      ;;
    --task_type)
      TASK_TYPE="$2"
      shift 2
      ;;
    --context_length)
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --max_target_length)
      MAX_TARGET_LENGTH="$2"
      shift 2
      ;;
    --filler_token)
      FILLER_TOKEN="$2"
      shift 2
      ;;
    --debug_repeat)
      DEBUG_REPEAT=true
      shift 1
      ;;
    --debug_gradients)
      DEBUG_GRADIENTS=true
      shift 1
      ;;
    --codebook_loss_weight) # New case for codebook_loss_weight
      CODEBOOK_LOSS_WEIGHT="$2"
      shift 2
      ;;
    --vq_use_argmax) # New case for vq_use_argmax
      VQ_USE_ARGMAX=true
      shift 1 # It's a flag, no second argument
      ;;
    --copy_test)
      COPY_TEST="$2"
      shift 2
      ;;
    --copy_test_samples)
      COPY_TEST_SAMPLES="$2"
      shift 2
      ;;
    --wiki)
      # Shortcut for running with wiki parameters (100/50/100)
      TASK_TYPE="wiki_continuation"
      CONTEXT_LENGTH=100
      VQ_REASON_LENGTH=50
      MAX_TARGET_LENGTH=100
      shift 1
      ;;
    --actor_hidden_layer_index) # New case
      ACTOR_HIDDEN_LAYER_INDEX="$2"
      shift 2
      ;;
    --normalize-reasoning-states)
      NORMALIZE_REASONING_STATES=true
      shift 1
      ;;
    --no-normalize-reasoning-states)
      NORMALIZE_REASONING_STATES=false
      shift 1
      ;;
    --vq_sequential_generation) # New case
      VQ_SEQUENTIAL_GENERATION=true
      shift 1
      ;;
    --checkpoint_frequency)
      CHECKPOINT_FREQUENCY="$2"
      shift 2
      ;;
    --plot_frequency)
      PLOT_FREQUENCY="$2"
      shift 2
      ;;
    --debug_answer_is_question) # New flag
      DEBUG_ANSWER_IS_QUESTION=true
      shift 1
      ;;
    --use_gumbel_softmax_vq) # New flag for Gumbel-Softmax VQ
      USE_GUMBEL_SOFTMAX_VQ=true
      shift 1
      ;;
    --gumbel_tau) # New argument for Gumbel-Softmax tau
      GUMBEL_TAU="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting minimal VQ training with the following parameters:"
echo "  Model: $MODEL"
echo "  Task: $TASK_TYPE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Batch size: $BATCH_SIZE"
echo "  Number of batches: $NUM_BATCHES"
echo "  VQ reasoning length: $VQ_REASON_LENGTH"
echo "  VQ sampling temperature: $VQ_SAMPLING_TEMP (higher = more diverse)"
echo "  Debug repeat datapoint: $DEBUG_REPEAT"
echo "  Debug gradients: $DEBUG_GRADIENTS"
echo "  Codebook Loss Weight: $CODEBOOK_LOSS_WEIGHT"
echo "  VQ Use Argmax: $VQ_USE_ARGMAX"
echo "  Actor Hidden Layer Index: $ACTOR_HIDDEN_LAYER_INDEX"
echo "  Normalize Reasoning States: $NORMALIZE_REASONING_STATES"
echo "  VQ Sequential Generation: $VQ_SEQUENTIAL_GENERATION"
echo "  Print frequency: $PRINT_FREQUENCY"
_CHECKPOINT_FREQ_MSG="  Checkpoint Frequency: Every $CHECKPOINT_FREQUENCY batches"
if [ "$CHECKPOINT_FREQUENCY" -le 0 ]; then
  _CHECKPOINT_FREQ_MSG="  Checkpoint Frequency: Disabled"
fi
echo "$_CHECKPOINT_FREQ_MSG"
_PLOT_FREQ_MSG="  Plot Frequency: Every $PLOT_FREQUENCY batches"
if [ "$PLOT_FREQUENCY" -le 0 ]; then
  _PLOT_FREQ_MSG="  Plot Frequency: Disabled"
fi
echo "$_PLOT_FREQ_MSG"
echo "  LoRA rank: $LORA_RANK"
echo "  Use Gumbel-Softmax VQ: $USE_GUMBEL_SOFTMAX_VQ"
echo "  Gumbel Tau: $GUMBEL_TAU"

_DEBUG_AIQ_MSG="  Debug Answer is Question: $DEBUG_ANSWER_IS_QUESTION"
echo "$_DEBUG_AIQ_MSG"

if [ "$COPY_TEST" -gt 0 ]; then
  echo "  Copy test: Every $COPY_TEST batches with $COPY_TEST_SAMPLES samples"
fi

if [ "$TASK_TYPE" == "wiki_continuation" ]; then
  echo "  Context length: $CONTEXT_LENGTH"
  echo "  Max target length: $MAX_TARGET_LENGTH"
fi

# Prepare debug flags
DEBUG_REPEAT_FLAG=""
if [ "$DEBUG_REPEAT" = true ]; then
  DEBUG_REPEAT_FLAG="--debug_repeat_datapoint"
fi

DEBUG_GRADIENTS_FLAG=""
if [ "$DEBUG_GRADIENTS" = true ]; then
  DEBUG_GRADIENTS_FLAG="--debug_gradients"
fi

VQ_ARGMAX_FLAG=""
if [ "$VQ_USE_ARGMAX" = true ]; then
  VQ_ARGMAX_FLAG="--vq_use_argmax"
fi

NORMALIZE_REASONING_STATES_FLAG=""
if [ "$NORMALIZE_REASONING_STATES" = true ]; then
  NORMALIZE_REASONING_STATES_FLAG="--normalize-reasoning-states"
else
  NORMALIZE_REASONING_STATES_FLAG="--no-normalize-reasoning-states"
fi

VQ_SEQUENTIAL_FLAG=""
if [ "$VQ_SEQUENTIAL_GENERATION" = true ]; then
  VQ_SEQUENTIAL_FLAG="--vq_sequential_generation"
fi

DEBUG_ANSWER_IS_QUESTION_FLAG=""
if [ "$DEBUG_ANSWER_IS_QUESTION" = true ]; then
  DEBUG_ANSWER_IS_QUESTION_FLAG="--debug_answer_is_question"
fi

USE_GUMBEL_SOFTMAX_VQ_FLAG=""
if [ "$USE_GUMBEL_SOFTMAX_VQ" = true ]; then
  USE_GUMBEL_SOFTMAX_VQ_FLAG="--use_gumbel_softmax_vq"
fi

# Run the training script
python3 vq_training_minimal.py \
  --model_name "$MODEL" \
  --task_type "$TASK_TYPE" \
  --learning_rate "$LEARNING_RATE" \
  --batch_size "$BATCH_SIZE" \
  --num_batches "$NUM_BATCHES" \
  --vq_reason_length "$VQ_REASON_LENGTH" \
  --vq_sampling_temp "$VQ_SAMPLING_TEMP" \
  --print_frequency "$PRINT_FREQUENCY" \
  --lora_rank "$LORA_RANK" \
  --context_length "$CONTEXT_LENGTH" \
  --max_target_length "$MAX_TARGET_LENGTH" \
  --codebook_loss_weight "$CODEBOOK_LOSS_WEIGHT" \
  --copy_test "$COPY_TEST" \
  --copy_test_samples "$COPY_TEST_SAMPLES" \
  --save_final_model \
  --actor_hidden_layer_index "$ACTOR_HIDDEN_LAYER_INDEX" \
  --checkpoint_frequency "$CHECKPOINT_FREQUENCY" \
  --plot_frequency "$PLOT_FREQUENCY" \
  $DEBUG_REPEAT_FLAG \
  $DEBUG_GRADIENTS_FLAG \
  $VQ_ARGMAX_FLAG \
  $NORMALIZE_REASONING_STATES_FLAG \
  $VQ_SEQUENTIAL_FLAG \
  $DEBUG_ANSWER_IS_QUESTION_FLAG \
  $USE_GUMBEL_SOFTMAX_VQ_FLAG \
  --gumbel_tau "$GUMBEL_TAU"

echo "Training complete!" 