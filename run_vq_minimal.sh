#!/bin/bash
# Run the minimal Vector Quantization training implementation

# Default parameters
MODEL="gpt2"
LEARNING_RATE=1e-4
BATCH_SIZE=700
NUM_BATCHES=100000
VQ_REASON_LENGTH=25
VQ_SAMPLING_TEMP=0 # Sampling temperature (higher = more diversity)
PRINT_FREQUENCY=100
LORA_RANK=0
TASK_TYPE="wiki_continuation"
CONTEXT_LENGTH=50
MAX_TARGET_LENGTH=50
#FILLER_TOKEN="<REASONING>"
DEBUG_REPEAT=false # Changed default for clarity
DEBUG_GRADIENTS=false
CODEBOOK_LOSS_WEIGHT=0.25
COPY_TEST=0  # Run copy test every N batches (0 to disable)
COPY_TEST_SAMPLES=3  # Number of samples for copy test
VQ_USE_ARGMAX=false # New default parameter for using argmax instead of sampling
ACTOR_HIDDEN_LAYER_INDEX=-2 
NORMALIZE_REASONING_STATES=true # New: Default to normalizing actor reasoning states
VQ_SEQUENTIAL_GENERATION=false # New default: sequential generation for VQ
CHECKPOINT_FREQUENCY=1000 # Default checkpoint frequency
PLOT_FREQUENCY=200 # New: Default plot frequency
DEBUG_ANSWER_IS_QUESTION=false # New: Default for answer is question debug
USE_GUMBEL_SOFTMAX_VQ=true # New: Use Gumbel-Softmax VQ
GUMBEL_TAU=2.0              # New: Tau for Gumbel-Softmax
USE_8BIT_ADAM=true           # New: Use 8-bit Adam
RESUME_CHECKPOINT_PATH="" # New: For resume path

# Parse command line arguments
PYTHON_SCRIPT_ARGS=() # Initialize as an array

show_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --model <string>                  Hugging Face model name (default: $MODEL)"
  echo "  --lr <float>                      Learning rate (default: $LEARNING_RATE)"
  echo "  --batch_size <int>                Batch size (default: $BATCH_SIZE)"
  echo "  --num_batches <int>               Number of batches to train (default: $NUM_BATCHES)"
  echo "  --vq_reason_length <int>          Length of VQ reasoning (default: $VQ_REASON_LENGTH)"
  echo "  --vq_sampling_temp <float>          VQ sampling temperature (default: $VQ_SAMPLING_TEMP)"
  echo "  --print_frequency <int>           Print frequency (default: $PRINT_FREQUENCY)"
  echo "  --lora_rank <int>                 LoRA rank, 0 to disable (default: $LORA_RANK)"
  echo "  --task_type <string>              Task type (default: $TASK_TYPE)"
  echo "  --context_length <int>            Context length (default: $CONTEXT_LENGTH)"
  echo "  --max_target_length <int>         Max target length (default: $MAX_TARGET_LENGTH)"
  echo "  --debug_repeat                    Enable debug repeat datapoint (default: $DEBUG_REPEAT)"
  echo "  --debug_gradients                 Enable debug gradients (default: $DEBUG_GRADIENTS)"
  echo "  --codebook_loss_weight <float>    Codebook loss weight (default: $CODEBOOK_LOSS_WEIGHT)"
  echo "  --vq_use_argmax                   Use argmax for VQ (default: $VQ_USE_ARGMAX)"
  echo "  --copy_test <int>                 Copy test frequency (default: $COPY_TEST)"
  echo "  --copy_test_samples <int>         Copy test samples (default: $COPY_TEST_SAMPLES)"
  echo "  --actor_hidden_layer_index <int>  Actor hidden layer index (default: $ACTOR_HIDDEN_LAYER_INDEX)"
  echo "  --[no-]normalize-reasoning-states Normalize reasoning states (default: $NORMALIZE_REASONING_STATES)"
  echo "  --vq_sequential_generation        Enable sequential VQ generation (default: $VQ_SEQUENTIAL_GENERATION)"
  echo "  --checkpoint_frequency <int>      Checkpoint frequency (default: $CHECKPOINT_FREQUENCY)"
  echo "  --plot_frequency <int>            Plot frequency (default: $PLOT_FREQUENCY)"
  echo "  --debug_answer_is_question        Set answer as question for debug (default: $DEBUG_ANSWER_IS_QUESTION)"
  echo "  --use_gumbel_softmax_vq           Use Gumbel-Softmax VQ (default: $USE_GUMBEL_SOFTMAX_VQ)"
  echo "  --gumbel_tau <float>              Tau for Gumbel-Softmax (default: $GUMBEL_TAU)"
  echo "  --use_8bit_adam                   Enable 8-bit Adam optimizer (default: $USE_8BIT_ADAM)"
  echo "  --resume <path>                   Path to checkpoint directory to resume from (e.g., results/task/timestamp/checkpoint_N)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --lr) LEARNING_RATE="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_batches) NUM_BATCHES="$2"; shift 2 ;;
    --vq_reason_length) VQ_REASON_LENGTH="$2"; shift 2 ;;
    --vq_sampling_temp) VQ_SAMPLING_TEMP="$2"; shift 2 ;;
    --print_frequency) PRINT_FREQUENCY="$2"; shift 2 ;;
    --lora_rank) LORA_RANK="$2"; shift 2 ;;
    --task_type) TASK_TYPE="$2"; shift 2 ;;
    --context_length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --max_target_length) MAX_TARGET_LENGTH="$2"; shift 2 ;;
    --debug_repeat) DEBUG_REPEAT=true; shift ;;
    --debug_gradients) DEBUG_GRADIENTS=true; shift ;;
    --codebook_loss_weight) CODEBOOK_LOSS_WEIGHT="$2"; shift 2 ;;
    --vq_use_argmax) VQ_USE_ARGMAX=true; shift ;;
    --copy_test) COPY_TEST="$2"; shift 2 ;;
    --copy_test_samples) COPY_TEST_SAMPLES="$2"; shift 2 ;;
    --actor_hidden_layer_index) ACTOR_HIDDEN_LAYER_INDEX="$2"; shift 2 ;;
    --normalize-reasoning-states) NORMALIZE_REASONING_STATES=true; shift ;;
    --no-normalize-reasoning-states) NORMALIZE_REASONING_STATES=false; shift ;;
    --vq_sequential_generation) VQ_SEQUENTIAL_GENERATION=true; shift ;;
    --checkpoint_frequency) CHECKPOINT_FREQUENCY="$2"; shift 2 ;;
    --plot_frequency) PLOT_FREQUENCY="$2"; shift 2 ;;
    --debug_answer_is_question) DEBUG_ANSWER_IS_QUESTION=true; shift ;;
    --use_gumbel_softmax_vq) USE_GUMBEL_SOFTMAX_VQ=true; shift ;;
    --gumbel_tau) GUMBEL_TAU="$2"; shift 2 ;;
    --use_8bit_adam) USE_8BIT_ADAM=true; shift ;;
    --resume) RESUME_CHECKPOINT_PATH="$2"; shift 2 ;; # Added resume parsing
    --help|-h) show_help ;;
    *) echo "Unknown option: $1"; show_help ;;
  esac
done

# Add parameters to PYTHON_SCRIPT_ARGS array
PYTHON_SCRIPT_ARGS+=(
  "--model_name" "$MODEL"
  "--task_type" "$TASK_TYPE"
  "--learning_rate" "$LEARNING_RATE"
  "--batch_size" "$BATCH_SIZE"
  "--num_batches" "$NUM_BATCHES"
  "--vq_reason_length" "$VQ_REASON_LENGTH"
  "--vq_sampling_temp" "$VQ_SAMPLING_TEMP"
  "--print_frequency" "$PRINT_FREQUENCY"
  "--lora_rank" "$LORA_RANK"
  "--context_length" "$CONTEXT_LENGTH"
  "--max_target_length" "$MAX_TARGET_LENGTH"
  "--codebook_loss_weight" "$CODEBOOK_LOSS_WEIGHT"
  "--copy_test" "$COPY_TEST"
  "--copy_test_samples" "$COPY_TEST_SAMPLES"
  "--actor_hidden_layer_index" "$ACTOR_HIDDEN_LAYER_INDEX"
  "--checkpoint_frequency" "$CHECKPOINT_FREQUENCY"
  "--plot_frequency" "$PLOT_FREQUENCY"
  "--gumbel_tau" "$GUMBEL_TAU"
  "--save_final_model" # This is a store_true flag, always present
)

if [ "$DEBUG_REPEAT" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--debug_repeat_datapoint")
fi
if [ "$DEBUG_GRADIENTS" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--debug_gradients")
fi
if [ "$VQ_USE_ARGMAX" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--vq_use_argmax")
fi
if [ "$NORMALIZE_REASONING_STATES" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--normalize-reasoning-states")
else
  PYTHON_SCRIPT_ARGS+=("--no-normalize-reasoning-states")
fi
if [ "$VQ_SEQUENTIAL_GENERATION" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--vq_sequential_generation")
fi
if [ "$DEBUG_ANSWER_IS_QUESTION" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--debug_answer_is_question")
fi
if [ "$USE_GUMBEL_SOFTMAX_VQ" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--use_gumbel_softmax_vq")
fi
if [ "$USE_8BIT_ADAM" = true ]; then
  PYTHON_SCRIPT_ARGS+=("--use_8bit_adam")
fi

# Handle resume functionality AFTER all other args are set
if [ -n "$RESUME_CHECKPOINT_PATH" ]; then
  PYTHON_SCRIPT_ARGS+=("--resume_from_checkpoint" "$RESUME_CHECKPOINT_PATH")
  echo "Resuming from checkpoint: $RESUME_CHECKPOINT_PATH"
fi

# Ensure base results directory exists
mkdir -p results 
# The python script itself now handles the output_dir creation/logging 
# within the correct (original or new) directory based on resume state.

# Construct the command
CMD="python3 vq_training_minimal.py ${PYTHON_SCRIPT_ARGS[@]}"

# Run the training script
echo "Running command:"
echo "$CMD"
eval "$CMD"

echo "Training complete!" 