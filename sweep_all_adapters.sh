#!/bin/bash
# Sweep over ALL adapter checkpoints with small sample sizes
# This creates a performance profile over training to identify best checkpoints

set -e

RESULTS_DIR="/root/MarkovianTraining/results"
STRIDE=20  # Evaluate every 20th example
NUM_SAMPLES=100  # Small sample for quick evaluation (use 0 for unlimited)
BATCH_SIZE=32  # Larger batch size for better GPU utilization

# Parse arguments
EXCLUDE_TASKS=""
INCLUDE_TASK=""
USE_HAIKU=false
PROMPT_VARIANT="default"
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            INCLUDE_TASK="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_TASKS="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --haiku)
            USE_HAIKU=true
            shift
            ;;
        --prompt-variant)
            PROMPT_VARIANT="$2"
            if [[ ! "$PROMPT_VARIANT" =~ ^(default|letter|letter_strict)$ ]]; then
                echo "Error: Invalid prompt variant '$PROMPT_VARIANT'"
                echo "Valid options: default, letter, letter_strict"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task TASK             Evaluate only this specific task (e.g., gsm8k, mmlu, arc)"
            echo "  --exclude TASKS         Comma-separated list of tasks to skip (e.g., mmlu,mathqa)"
            echo "  --stride N              Evaluate every Nth example (default: 20)"
            echo "  --samples N             Number of samples per evaluation (default: 100, use 0 for unlimited)"
            echo "  --batch-size N          Batch size for evaluation (default: 32)"
            echo "  --prompt-variant TYPE   Answer prompt variant for MCQ tasks (default|letter|letter_strict, default: default)"
            echo "  --haiku                 Enable Haiku LLM-based extraction validation (requires ANTHROPIC_API_KEY)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --task gsm8k --samples 0 --batch-size 128  # Full GSM8K evaluation"
            echo "  $0 --exclude mmlu,mathqa --stride 10"
            echo "  $0 --haiku --samples 50  # Use Haiku validation on small sample"
            echo "  $0 --prompt-variant letter --batch-size 64  # Use letter prompt variant"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert exclude list to array
IFS=',' read -ra EXCLUDE_ARRAY <<< "$EXCLUDE_TASKS"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "========================================================================"
echo "ADAPTER SWEEP - All Checkpoints Profile"
echo "========================================================================"
echo "Stride: $STRIDE | Samples: $NUM_SAMPLES | Batch Size: $BATCH_SIZE"
if [[ -n "$INCLUDE_TASK" ]]; then
    echo "Task: $INCLUDE_TASK (only)"
fi
if [[ "$PROMPT_VARIANT" != "default" ]]; then
    echo "Prompt Variant: $PROMPT_VARIANT"
fi
if [[ "$USE_HAIKU" == true ]]; then
    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        echo -e "Haiku validation: ${GREEN}ENABLED${NC}"
    else
        echo -e "${RED}ERROR: --haiku flag specified but ANTHROPIC_API_KEY is not set${NC}"
        echo -e "${RED}Please set the ANTHROPIC_API_KEY environment variable or remove the --haiku flag${NC}"
        exit 1
    fi
fi
if [[ -n "$EXCLUDE_TASKS" ]]; then
    echo "Excluding: $EXCLUDE_TASKS"
fi
echo ""

TOTAL_EVALS=0
SUCCESSFUL=0
FAILED=0

# Create output directory for sweep results
SWEEP_DIR="$RESULTS_DIR/adapter_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

echo "Results will be saved to: $SWEEP_DIR"
echo ""

# Function to evaluate a single adapter
evaluate_adapter() {
    local task=$1
    local run_dir=$2
    local adapter=$3
    local adapter_name=$(basename "$adapter")
    local run_name=$(basename "$run_dir")
    
    TOTAL_EVALS=$((TOTAL_EVALS + 1))
    
    # Build command
    CMD="PYTHONPATH=/root/MarkovianTraining/src python -m evaluation \
        --task_type $task \
        --model_path $adapter \
        --stride $STRIDE \
        --batch_size $BATCH_SIZE"
    
    # Only add num_samples if > 0 (0 means unlimited)
    if [[ $NUM_SAMPLES -gt 0 ]]; then
        CMD="$CMD --num_samples $NUM_SAMPLES"
    fi
    
    # Add prompt variant if not default
    if [[ "$PROMPT_VARIANT" != "default" ]]; then
        CMD="$CMD --answer_prompt_variant $PROMPT_VARIANT"
    fi
    
    # Add Haiku if requested
    if [[ "$USE_HAIKU" == true ]]; then
        CMD="$CMD --haiku_metric"
    fi
    
    # Task-specific adjustments
    if [[ "$task" == "mmlu" ]]; then
        CMD="$CMD --num_samples 50"  # MMLU is slow, use fewer samples
    fi
    
    # Suppress most output, just get the accuracy
    if output=$(eval "$CMD" 2>&1); then
        # Extract accuracy from output - multiple formats
        # Format 1: 'XX.XX%' (with quotes, on separate line)
        accuracy=$(echo "$output" | grep -oP "'\d+\.?\d+%'" | grep -oP "\d+\.?\d+" | head -1)
        
        if [[ -z "$accuracy" ]]; then
            # Format 2: "Accuracy: XX.XX%" or "accuracy: XX.XX%"
            accuracy=$(echo "$output" | grep -oP "(?:Accuracy|accuracy)[:\s]+(\d+\.?\d*)%" | grep -oP "\d+\.?\d+" | head -1)
        fi
        
        if [[ -z "$accuracy" ]]; then
            # Format 3: Decimal format (e.g., "0.52")
            accuracy=$(echo "$output" | grep -oP "(?:Accuracy|accuracy)[:\s]+(0\.\d+)" | grep -oP "0\.\d+" | head -1)
            if [[ -n "$accuracy" ]]; then
                # Convert decimal to percentage (0.52 -> 52)
                accuracy=$(echo "$accuracy * 100" | bc -l | xargs printf "%.2f")
            fi
        fi
        
        # Word boundary accuracy (for MCQ tasks)
        accuracy_wb=$(echo "$output" | grep -i "word boundary" | grep -oP "\d+\.?\d+" | head -1)
        
        # Haiku accuracy (if enabled)
        haiku_accuracy=""
        if [[ "$USE_HAIKU" == true ]]; then
            haiku_accuracy=$(echo "$output" | grep -i "haiku.*accuracy" | grep -oP "\d+\.?\d+" | head -1)
        fi
        
        if [[ -n "$accuracy" ]]; then
            SUCCESSFUL=$((SUCCESSFUL + 1))
            display_msg="${GREEN}✓${NC} $adapter_name: ${CYAN}${accuracy}%${NC}"
            if [[ -n "$haiku_accuracy" ]]; then
                display_msg="$display_msg (Haiku: ${haiku_accuracy}%)"
            fi
            echo -e "$display_msg"
            
            # Save to sweep results file
            echo "$task,$run_name,$adapter_name,$accuracy,$accuracy_wb,$haiku_accuracy" >> "$SWEEP_DIR/${task}_sweep.csv"
        else
            FAILED=$((FAILED + 1))
            echo -e "${RED}✗${NC} $adapter_name: No accuracy found"
        fi
    else
        FAILED=$((FAILED + 1))
        echo -e "${RED}✗${NC} $adapter_name: Evaluation failed"
    fi
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# Process each task
for task_dir in "$RESULTS_DIR"/*; do
    if [[ ! -d "$task_dir" ]]; then
        continue
    fi
    
    task=$(basename "$task_dir")
    
    # Skip wiki, sweep directories, and non-task directories
    if [[ "$task" =~ ^wiki || "$task" =~ ^adapter_sweep || "$task" == "keep.txt" || "$task" == "arithmetic" ]]; then
        continue
    fi
    
    # If --task is specified, only process that task
    if [[ -n "$INCLUDE_TASK" ]]; then
        if [[ "$task" != "$INCLUDE_TASK" ]]; then
            continue
        fi
    fi
    
    # Skip excluded tasks (only if --task is not specified)
    if [[ -z "$INCLUDE_TASK" ]]; then
        for excluded in "${EXCLUDE_ARRAY[@]}"; do
            if [[ "$task" == "$excluded" ]]; then
                echo -e "${YELLOW}Skipping excluded task: $task${NC}"
                continue 2  # Continue outer loop
            fi
        done
    fi
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}Task: $task${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    # Create CSV header for this task
    echo "task,run,adapter,accuracy,accuracy_wb,haiku_accuracy" > "$SWEEP_DIR/${task}_sweep.csv"
    
    # Find all timestamp directories
    for run_dir in "$task_dir"/*/; do
        if [[ ! -d "$run_dir" ]]; then
            continue
        fi
        
        run_name=$(basename "$run_dir")
        echo ""
        echo -e "${YELLOW}Run: $run_name${NC}"
        
        # Find ALL adapter checkpoints and sort them numerically
        adapters=($(find "$run_dir" -maxdepth 1 -type d -name "adapter_*" | sort -V))
        
        if [[ ${#adapters[@]} -eq 0 ]]; then
            echo "  No adapters found"
            continue
        fi
        
        echo "  Found ${#adapters[@]} adapters"
        
        # First evaluate base model (adapter_0) for baseline comparison
        echo -n "  "
        echo -e "${CYAN}[Baseline]${NC} adapter_0 (no adapter)"
        
        # Determine base model path from the first adapter's config
        if [[ ${#adapters[@]} -gt 0 ]]; then
            # Read base_model_name_or_path from first adapter's config
            first_adapter="${adapters[0]}"
            base_model=$(grep -oP '"base_model_name_or_path":\s*"\K[^"]+' "$first_adapter/adapter_config.json" 2>/dev/null || echo "")
            
            if [[ -n "$base_model" ]]; then
                # Evaluate with base model
                echo -n "  "
                TOTAL_EVALS=$((TOTAL_EVALS + 1))
                
                CMD="PYTHONPATH=/root/MarkovianTraining/src python -m evaluation \
                    --task_type $task \
                    --model_path $base_model \
                    --use_base_model \
                    --stride $STRIDE \
                    --batch_size $BATCH_SIZE"
                
                # Only add num_samples if > 0 (0 means unlimited)
                if [[ $NUM_SAMPLES -gt 0 ]]; then
                    CMD="$CMD --num_samples $NUM_SAMPLES"
                fi
                
                # Add prompt variant if not default
                if [[ "$PROMPT_VARIANT" != "default" ]]; then
                    CMD="$CMD --answer_prompt_variant $PROMPT_VARIANT"
                fi
                
                # Add Haiku if requested
                if [[ "$USE_HAIKU" == true ]]; then
                    CMD="$CMD --haiku_metric"
                fi
                
                # Task-specific adjustments
                if [[ "$task" == "mmlu" ]]; then
                    CMD="$CMD --num_samples 50"
                fi
                
                if output=$(eval "$CMD" 2>&1); then
                    # Extract accuracy - same logic as evaluate_adapter
                    accuracy=$(echo "$output" | grep -oP "'\d+\.?\d+%'" | grep -oP "\d+\.?\d+" | head -1)
                    
                    if [[ -z "$accuracy" ]]; then
                        accuracy=$(echo "$output" | grep -oP "(?:Accuracy|accuracy)[:\s]+(\d+\.?\d*)%" | grep -oP "\d+\.?\d+" | head -1)
                    fi
                    
                    if [[ -z "$accuracy" ]]; then
                        accuracy=$(echo "$output" | grep -oP "(?:Accuracy|accuracy)[:\s]+(0\.\d+)" | grep -oP "0\.\d+" | head -1)
                        if [[ -n "$accuracy" ]]; then
                            accuracy=$(echo "$accuracy * 100" | bc -l | xargs printf "%.2f")
                        fi
                    fi
                    
                    accuracy_wb=$(echo "$output" | grep -i "word boundary" | grep -oP "\d+\.?\d+" | head -1)
                    
                    haiku_accuracy=""
                    if [[ "$USE_HAIKU" == true ]]; then
                        haiku_accuracy=$(echo "$output" | grep -i "haiku.*accuracy" | grep -oP "\d+\.?\d+" | head -1)
                    fi
                    
                    if [[ -n "$accuracy" ]]; then
                        SUCCESSFUL=$((SUCCESSFUL + 1))
                        display_msg="${GREEN}✓${NC} adapter_0: ${CYAN}${accuracy}%${NC}"
                        if [[ -n "$haiku_accuracy" ]]; then
                            display_msg="$display_msg (Haiku: ${haiku_accuracy}%)"
                        fi
                        echo -e "$display_msg"
                        echo "$task,$run_name,adapter_0,$accuracy,$accuracy_wb,$haiku_accuracy" >> "$SWEEP_DIR/${task}_sweep.csv"
                    else
                        FAILED=$((FAILED + 1))
                        echo -e "${RED}✗${NC} adapter_0: No accuracy found"
                    fi
                else
                    FAILED=$((FAILED + 1))
                    echo -e "${RED}✗${NC} adapter_0: Evaluation failed"
                fi
                
                # Clear GPU cache
                python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            else
                echo -e "  ${YELLOW}⚠${NC} Could not find base model path, skipping adapter_0"
            fi
        fi
        
        # Evaluate each adapter
        for adapter in "${adapters[@]}"; do
            echo -n "  "
            evaluate_adapter "$task" "$run_dir" "$adapter"
        done
    done
done

# Generate summary report
echo ""
echo "========================================================================"
echo -e "${BLUE}SWEEP SUMMARY${NC}"
echo "========================================================================"
echo "Total evaluations: $TOTAL_EVALS"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""
echo "Results saved to: $SWEEP_DIR"
echo ""

# Create summary plots with Python
cat > "$SWEEP_DIR/plot_sweep.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

sweep_dir = os.path.dirname(os.path.abspath(__file__))
csv_files = glob.glob(os.path.join(sweep_dir, "*_sweep.csv"))

for csv_file in csv_files:
    task = os.path.basename(csv_file).replace("_sweep.csv", "")
    
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue
        
        # Extract batch number from adapter name
        df['batch'] = df['adapter'].str.extract(r'adapter_(\d+)').astype(int)
        
        # Plot for each run
        runs = df['run'].unique()
        
        fig, axes = plt.subplots(len(runs), 1, figsize=(12, 4*len(runs)))
        if len(runs) == 1:
            axes = [axes]
        
        for idx, run in enumerate(runs):
            run_data = df[df['run'] == run].sort_values('batch')
            
            ax = axes[idx]
            ax.plot(run_data['batch'], run_data['accuracy'], 'o-', label='Primary', linewidth=2, markersize=8)
            
            # Add word boundary if available
            if 'accuracy_wb' in run_data.columns and run_data['accuracy_wb'].notna().any():
                ax.plot(run_data['batch'], run_data['accuracy_wb'], 's-', label='Word Boundary', linewidth=2, markersize=6)
            
            # Highlight best checkpoint
            best_idx = run_data['accuracy'].idxmax()
            best_batch = run_data.loc[best_idx, 'batch']
            best_acc = run_data.loc[best_idx, 'accuracy']
            ax.axvline(best_batch, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_batch}')
            ax.plot(best_batch, best_acc, 'r*', markersize=20)
            
            ax.set_xlabel('Training Batch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{task.upper()} - {run}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(sweep_dir, f'{task}_sweep_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Created plot: {task}_sweep_plot.png")
        
        # Print best checkpoints
        print(f"\n{task.upper()} - Best Checkpoints:")
        for run in runs:
            run_data = df[df['run'] == run].sort_values('batch')
            best_idx = run_data['accuracy'].idxmax()
            best_row = run_data.loc[best_idx]
            print(f"  {run}: adapter_{int(best_row['batch'])} - {best_row['accuracy']:.2f}%")
        
    except Exception as e:
        print(f"Error processing {task}: {e}")

print("\nAll plots generated!")
EOF

chmod +x "$SWEEP_DIR/plot_sweep.py"

echo "Generating plots..."
cd "$SWEEP_DIR" && python plot_sweep.py

echo ""
echo "========================================================================"
echo "SWEEP COMPLETE"
echo "========================================================================"
echo "View results in: $SWEEP_DIR"
echo "- CSV files: *_sweep.csv"
echo "- Plots: *_sweep_plot.png"

