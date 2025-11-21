#!/usr/bin/env bash

# Sync only perturbation analysis artifacts to S3.
# This avoids uploading entire run directories when we only
# need perturbation result JSON/plots and comparison outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--bucket S3_URI] [--dry-run]

Options:
  --bucket   Destination bucket/prefix (default: \$PERTURB_S3_BUCKET or s3://scottviteri/results)
  --dry-run  Pass --dryrun to aws s3 sync for verification.
EOF
}

BUCKET="${PERTURB_S3_BUCKET:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$BUCKET" ]]; then
    BUCKET="s3://scottviteri/results"
else
    # Ensure bucket path starts with s3:// and points to /results
    if [[ ! "$BUCKET" =~ ^s3:// ]]; then
        BUCKET="s3://${BUCKET}"
    fi
    BUCKET="${BUCKET%/}"
    if [[ ! "$BUCKET" =~ /results$ ]]; then
        BUCKET="${BUCKET}/results"
    fi
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Results directory not found at $RESULTS_DIR" >&2
    exit 1
fi

CMD=(aws s3 sync "$RESULTS_DIR" "$BUCKET" --exclude "*"
    --include "*/perturbation_results_*.json"
    --include "*/perturbation_results_*_plot.png"
    --include "*/perturbation_results_*_debug.png"
    --include "*/markovian_comparison_accuracy/*.json"
    --include "*/markovian_comparison_accuracy/*.png")

if [[ "$DRY_RUN" == true ]]; then
    CMD+=(--dryrun)
fi

echo "Syncing perturbation artifacts to $BUCKET"
echo "Command: ${CMD[*]}"
"${CMD[@]}"


