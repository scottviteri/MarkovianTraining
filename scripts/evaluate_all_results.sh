#!/bin/bash
# Comprehensive evaluation script for all results
# This script evaluates all checkpoints in the results folder
# Usage: ./scripts/evaluate_all_results.sh [--light]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
PYTHONPATH_VALUE="$REPO_ROOT/src"

LIGHT_MODE=false
if [[ "$1" == "--light" ]]; then
    LIGHT_MODE=true
    echo "Running in LIGHT mode (stride=10, limited samples)"
else
    echo "Running in FULL evaluation mode"
fi

RESULTS_DIR="$REPO_ROOT/results"
TOTAL_RUNS=0
SUCCESSFUL=0
FAILED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "COMPREHENSIVE EVALUATION OF ALL RESULTS"
echo "========================================================================"
echo ""

# Function to evaluate a single run
evaluate_run() {
    local task=$1
    local run_dir=$2
    local adapter=$3
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    
    echo ""
    echo -e "${BLUE}[$TOTAL_RUNS] Evaluating:${NC} $task / $(basename $run_dir) / $(basename $adapter)"
    
    # Build command
    CMD="PYTHONPATH=${PYTHONPATH_VALUE} python -m evaluation --task_type $task --model_path $adapter"
    
    # Add light mode parameters
    if [[ "$LIGHT_MODE" == true ]]; then
        CMD="$CMD --stride 10"
        if [[ "$task" == "mmlu" ]]; then
            CMD="$CMD --num_samples 50"
        elif [[ "$task" == "gsm8k" ]]; then
            CMD="$CMD --num_samples 100"
        elif [[ "$task" == "svamp" || "$task" == "aqua" || "$task" == "mathqa" || "$task" == "arc" ]]; then
            CMD="$CMD --num_samples 50"
        fi
    else
        # Full evaluation parameters
        if [[ "$task" == "mmlu" ]]; then
            CMD="$CMD --num_samples 500"
        fi
    fi
    
    # Run evaluation
    if eval "$CMD" 2>&1 | tail -5; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
        echo -e "${GREEN}✓ Success${NC}"
    else
        FAILED=$((FAILED + 1))
        echo -e "${RED}✗ Failed${NC}"
    fi
    
    # Clear GPU cache between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# Find and evaluate all runs
for task_dir in "$RESULTS_DIR"/*; do
    if [[ ! -d "$task_dir" ]]; then
        continue
    fi
    
    task=$(basename "$task_dir")
    
    # Skip wiki tasks (not about exact accuracy)
    if [[ "$task" == "wiki_continuation" || "$task" == "wiki_compression" || "$task" == "wiki_"* ]]; then
        echo -e "${YELLOW}Skipping $task (wiki task)${NC}"
        continue
    fi
    
    # Skip non-task directories
    if [[ "$task" == "keep.txt" || "$task" == "arithmetic" ]]; then
        continue
    fi
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}Task: $task${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    # Find all timestamp directories
    for run_dir in "$task_dir"/*/; do
        if [[ ! -d "$run_dir" ]]; then
            continue
        fi
        
        # Find latest adapter checkpoint
        latest_adapter=$(find "$run_dir" -maxdepth 1 -type d -name "adapter_*" | sort -V | tail -1)
        
        if [[ -n "$latest_adapter" ]]; then
            evaluate_run "$task" "$run_dir" "$latest_adapter"
        fi
    done
done

# Summary
echo ""
echo "========================================================================"
echo -e "${BLUE}EVALUATION SUMMARY${NC}"
echo "========================================================================"
echo -e "Total runs evaluated: $TOTAL_RUNS"
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo "========================================================================"
echo ""

if [[ "$LIGHT_MODE" == true ]]; then
    echo -e "${YELLOW}This was a LIGHT evaluation (stride=10, limited samples)${NC}"
    echo -e "${YELLOW}Run without --light flag for full evaluation${NC}"
fi

