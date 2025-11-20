#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Define the base directory where your results are stored
BASE_DIR="results/GSM8KFinalRuns"

# Define the runs you want to include (e.g., results_1, results_2, results_3)
RUNS=("results_1" "results_2" "results_3")

# Define the perturbation types you want to analyze
PERTURB_TYPES=("delete" "truncate_front" "truncate_back" "char_replace")

# Create a directory for collated perturbation results
COLLATED_DIR="${BASE_DIR}/collated_perturbations"
mkdir -p "${COLLATED_DIR}"

# Step 1: Generate perturbation results for each run and perturbation type
echo "Generating perturbation results for each run and perturbation type..."
for RUN in "${RUNS[@]}"; do
  echo "Processing ${RUN}..."
  for PERTURB in "${PERTURB_TYPES[@]}"; do
    echo " - Perturbation: ${PERTURB}"
    python src/perturbation_analysis.py \
      --log_file "${BASE_DIR}/${RUN}/log.jsonl" \
      --perturb "${PERTURB}" \
      --window_size 200
  done
done

# Step 2: Collate perturbation results across runs for each perturbation type
echo "Collating perturbation results across runs..."
for PERTURB in "${PERTURB_TYPES[@]}"; do
  echo "Collating results for ${PERTURB}..."
  PERTURB_FILES=()
  for RUN in "${RUNS[@]}"; do
    PERTURB_FILE="${BASE_DIR}/${RUN}/perturbation_results_${PERTURB}.json"
    if [ -f "${PERTURB_FILE}" ]; then
      PERTURB_FILES+=("${PERTURB_FILE}")
    else
      echo "Warning: ${PERTURB_FILE} does not exist."
    fi
  done
  # Specify the output directory for the collated results
  OUTPUT_DIR="${COLLATED_DIR}/${PERTURB}"
  mkdir -p "${OUTPUT_DIR}"
  # Collate the perturbation results
  python src/perturbation_analysis.py \
    --collate "${PERTURB_FILES[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --plot_only
done

# (Optional) If you moved all perturbation results into collated_perturbations, skip moving files

# Step 3: Generate individual plots for each perturbation type
echo "Generating individual plots for each perturbation type..."
for PERTURB in "${PERTURB_TYPES[@]}"; do
  echo "Plotting results for ${PERTURB}..."
  python src/perturbation_analysis.py \
    --log_file "${COLLATED_DIR}/${PERTURB}/" \
    --plot_only \
    --perturb "${PERTURB}" \
    --window_size 200 \
    --font_size 14 \
    --legend_font_size 12
done

# Step 4: Generate combined plot for multiple perturbations
echo "Generating combined plot for multiple perturbations..."
python src/perturbation_analysis.py \
  --log_file "${COLLATED_DIR}/" \
  --plot_only \
  --perturb "${PERTURB_TYPES[@]}" \
  --plot_multiple_perturbations \
  --window_size 200 \
  --font_size 14 \
  --legend_font_size 12

# Step 5: (Optional) Collate cross-model evaluation results
# Define evaluation result files (replace with your actual files)
EVAL_FILES=(
  "${BASE_DIR}/results_1/evaluation_results.jsonl"
  "${BASE_DIR}/results_2/evaluation_results.jsonl"
  "${BASE_DIR}/results_3/evaluation_results.jsonl"
)

# (Optional) Create a directory for collated evaluation results
EVAL_COLLATED_DIR="${BASE_DIR}/collated_evaluations"
mkdir -p "${EVAL_COLLATED_DIR}"

echo "Collating cross-model evaluation results..."
python src/evaluate_cross_model.py \
  --collate "${EVAL_FILES[@]}" \
  --output_dir "${EVAL_COLLATED_DIR}"

# Step 6: (Optional) Generate combined plot for multiple critics
echo "Generating combined plot for multiple critics..."
python src/evaluate_cross_model.py \
  --log_file "${EVAL_COLLATED_DIR}/" \
  --plot_only \
  --plot_multiple_critics \
  --window_size 200 \
  --font_size 14 \
  --legend_font_size 12

echo "All tasks completed."