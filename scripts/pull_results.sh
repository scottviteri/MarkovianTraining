#!/bin/bash

# Pull results from remote host to local ICLR2026Rebuttals folder
# Usage: ./scripts/pull_results.sh --source <host[:dataset[:timestamp_or_index]]> [--parallel N] [--all]
# Examples:
#   ./scripts/pull_results.sh --source right2
#   ./scripts/pull_results.sh --source right2:mmlu --parallel 4
#   ./scripts/pull_results.sh --source right2:mmlu --all --parallel 4  # Pull all mmlu runs (4 at once)
#   ./scripts/pull_results.sh --source right2:mmlu:-1
#   ./scripts/pull_results.sh --source right2:mmlu:20251116_191617

# Note: Don't use 'set -e' because we want to handle rsync failures and retry

usage() {
    echo "Usage: $0 --source <host[:dataset[:timestamp_or_index]]> [--parallel N] [--all]"
    echo ""
    echo "Examples:"
    echo "  $0 --source right2"
    echo "  $0 --source right2:mmlu --parallel 4"
    echo "  $0 --source right2:mmlu --all              # Pull all mmlu runs (sequential)"
    echo "  $0 --source right2:mmlu --all --parallel 4 # Pull all mmlu runs (4 at a time)"
    echo "  $0 --source right2:mmlu:-1                 # 2nd most recent"
    echo "  $0 --source right2:mmlu:20251116_191617"
    echo ""
    echo "Options:"
    echo "  --parallel N    Transfer N subdirectories simultaneously (single run)"
    echo "                  or N runs simultaneously (with --all) (default: 1)"
    echo "  --all           Pull all runs matching the spec (ignores timestamp/index)"
    echo ""
    echo "Timestamp/Index:"
    echo "  (omitted)   - Most recent (default)"
    echo "  -1          - 2nd most recent"
    echo "  -2          - 3rd most recent"
    echo "  20251116... - Specific timestamp"
    exit 1
}

# Parse arguments
SOURCE=""
PARALLEL=1
PULL_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --all)
            PULL_ALL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ -z "$SOURCE" ]; then
    echo "Error: --source is required"
    usage
fi

# Parse source (format: host[:dataset[:timestamp_or_index]])
IFS=':' read -r HOST DATASET TIMESTAMP_OR_INDEX <<< "$SOURCE"

if [ -z "$HOST" ]; then
    echo "Error: Source must include at least a hostname"
    usage
fi

LOCAL_BASE="/home/scottviteri/Projects/MarkovianTraining/ICLR2026Rebuttals/Logs"
REMOTE_BASE="/root/MarkovianTraining/results"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Handle --all flag
if [ "$PULL_ALL" = true ]; then
    if [ -z "$DATASET" ]; then
        echo -e "${RED}Error: --all requires a dataset to be specified${NC}"
        echo "Usage: $0 --source host:dataset --all"
        exit 1
    fi
    
    if [ -n "$TIMESTAMP_OR_INDEX" ]; then
        echo -e "${YELLOW}Warning: --all ignores timestamp/index${NC}"
    fi
    
    echo -e "${BLUE}Finding all $DATASET runs on $HOST${NC}"
    
    REMOTE_PATH="$REMOTE_BASE/$DATASET"
    
    # Get all directories sorted by modification time
    ALL_DIRS=$(ssh "$HOST" "find $REMOTE_PATH -maxdepth 1 -type d -name '2025*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2" 2>/dev/null)
    
    if [ -z "$ALL_DIRS" ]; then
        echo -e "${RED}No $DATASET runs found on $HOST${NC}"
        exit 1
    fi
    
    DIR_COUNT=$(echo "$ALL_DIRS" | wc -l)
    echo -e "${GREEN}Found $DIR_COUNT runs to pull${NC}"
    echo ""
    
    # Pull each directory
    SUCCESS=0
    FAILED=0
    CURRENT=0
    
    while IFS= read -r REMOTE_DIR; do
        CURRENT=$((CURRENT + 1))
        DIR_NAME=$(basename "$REMOTE_DIR")
        
        echo -e "${YELLOW}[$CURRENT/$DIR_COUNT] Pulling $DATASET/$DIR_NAME${NC}"
        
        LOCAL_DIR="$LOCAL_BASE/$HOST/$DATASET"
        mkdir -p "$LOCAL_DIR"
        
        # Check if already exists
        if [ -d "$LOCAL_DIR/$DIR_NAME" ]; then
            echo -e "${BLUE}  Already exists, syncing...${NC}"
        fi
        
        # Pull with single attempt (no retry loop in --all mode for speed)
        # Note: --parallel flag is not used in --all mode (pulls runs sequentially)
        rsync -avP --append-verify --timeout=60 "root@$HOST:$REMOTE_DIR/" "$LOCAL_DIR/$DIR_NAME/" 2>&1 | grep -E "receiving|total size|speedup|/$" || true
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo -e "${GREEN}  ✓ Complete${NC}"
            SUCCESS=$((SUCCESS + 1))
        else
            echo -e "${RED}  ✗ Failed${NC}"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done <<< "$ALL_DIRS"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Summary:${NC}"
    echo -e "  Total: $DIR_COUNT"
    echo -e "  ${GREEN}Success: $SUCCESS${NC}"
    echo -e "  ${RED}Failed: $FAILED${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
    
    exit 0
fi

# Check if TIMESTAMP_OR_INDEX is a negative number (index)
if [[ "$TIMESTAMP_OR_INDEX" =~ ^-[0-9]+$ ]]; then
    # Negative index provided
    if [ -z "$DATASET" ]; then
        echo -e "${RED}Error: Dataset type required when using index${NC}"
        echo "Usage: $0 --source $HOST:dataset:index"
        exit 1
    fi
    
    INDEX=${TIMESTAMP_OR_INDEX#-}  # Remove the minus sign to get absolute value
    INDEX=$((INDEX + 1))  # Convert: -1 becomes 2nd (line 2), -2 becomes 3rd (line 3), etc.
    
    echo -e "${BLUE}Finding ${INDEX}th most recent results on $HOST${NC}"
    echo -e "${BLUE}Dataset: $DATASET${NC}"
    
    REMOTE_PATH="$REMOTE_BASE/$DATASET"
    
    # Get list of directories sorted by recency, pick the Nth one
    MOST_RECENT=$(ssh "$HOST" "find $REMOTE_PATH -maxdepth 1 -type d -name '2025*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | sed -n '${INDEX}p' | cut -d' ' -f2" 2>/dev/null)
    
    if [ -z "$MOST_RECENT" ]; then
        echo -e "${RED}No directory found at index $TIMESTAMP_OR_INDEX (only $(ssh "$HOST" "find $REMOTE_PATH -maxdepth 1 -type d -name '2025*' 2>/dev/null | wc -l") directories available)${NC}"
        exit 1
    fi
    
    DIR_NAME=$(basename "$MOST_RECENT")
    DATASET_TYPE="$DATASET"
    
elif [ -n "$TIMESTAMP_OR_INDEX" ]; then
    # Specific timestamp provided
    if [ -z "$DATASET" ]; then
        echo -e "${RED}Error: Dataset type required when specifying timestamp${NC}"
        echo "Usage: $0 --source $HOST:dataset:timestamp"
        exit 1
    fi
    
    echo -e "${BLUE}Looking for specific run on $HOST${NC}"
    echo -e "${BLUE}Dataset: $DATASET, Timestamp: $TIMESTAMP_OR_INDEX${NC}"
    
    MOST_RECENT="$REMOTE_BASE/$DATASET/$TIMESTAMP_OR_INDEX"
    
    # Verify the directory exists
    if ! ssh "$HOST" "test -d $MOST_RECENT" 2>/dev/null; then
        echo -e "${RED}Directory not found: $MOST_RECENT${NC}"
        exit 1
    fi
    
    DIR_NAME="$TIMESTAMP_OR_INDEX"
    DATASET_TYPE="$DATASET"
else
    # Find most recent (index 0, default)
    echo -e "${BLUE}Finding most recent results on $HOST${NC}"
    
    # Build the remote path
    if [ -n "$DATASET" ]; then
        REMOTE_PATH="$REMOTE_BASE/$DATASET"
        echo -e "${BLUE}Looking for dataset: $DATASET${NC}"
    else
        REMOTE_PATH="$REMOTE_BASE"
        echo -e "${BLUE}Looking across all datasets${NC}"
    fi
    
    # Find most recent directory on remote host
    # List directories with timestamps, sort by modification time, get most recent
    MOST_RECENT=$(ssh "$HOST" "find $REMOTE_PATH -maxdepth 2 -type d -name '2025*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2" 2>/dev/null)
    
    if [ -z "$MOST_RECENT" ]; then
        echo -e "${RED}No results directories found on $HOST${NC}"
        exit 1
    fi
    
    # Extract just the directory name and dataset type
    DIR_NAME=$(basename "$MOST_RECENT")
    DATASET_TYPE=$(basename $(dirname "$MOST_RECENT"))
fi

echo -e "${GREEN}Found: $DATASET_TYPE/$DIR_NAME${NC}"

# Create local directory structure: Logs/hostname/dataset/timestamp
LOCAL_DIR="$LOCAL_BASE/$HOST/$DATASET_TYPE"
mkdir -p "$LOCAL_DIR"

# Pull the data using rsync with retry logic
echo -e "${BLUE}Pulling from $HOST:$MOST_RECENT${NC}"
echo -e "${BLUE}To: $LOCAL_DIR/$DIR_NAME${NC}"
echo -e "${YELLOW}Parallelism: ${NC}$PARALLEL"
echo ""

if [ $PARALLEL -eq 1 ]; then
    # Sequential mode - single rsync as before
    echo -e "${BLUE}Transferring (sequential mode)${NC}"
    
    START_TIME=$(date +%s)
    TIMEOUT=300
    ATTEMPT=0
    
    while true; do
        ATTEMPT=$((ATTEMPT + 1))
        ELAPSED=$(($(date +%s) - START_TIME))
        
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo -e "${RED}✗ Failed to pull results after 5 minutes${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}Attempt $ATTEMPT (${ELAPSED}s elapsed)${NC}"
        
        rsync -avP --append-verify --timeout=60 "root@$HOST:$MOST_RECENT/" "$LOCAL_DIR/$DIR_NAME/"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Successfully pulled $DATASET_TYPE/$DIR_NAME from $HOST${NC}"
            echo -e "${GREEN}  Location: $LOCAL_DIR/$DIR_NAME${NC}"
            exit 0
        fi
        
        echo -e "${RED}✗ Transfer failed, retrying immediately...${NC}"
    done
else
    # Parallel mode - transfer subdirectories in parallel
    echo -e "${BLUE}Transferring (parallel mode)${NC}"
    
    # First, get list of remote subdirectories
    REMOTE_SUBDIRS=$(ssh "$HOST" "find $MOST_RECENT -mindepth 1 -maxdepth 1 -type d -exec basename {} \;" 2>/dev/null)
    
    # Transfer root-level files first (non-recursive)
    echo -e "${BLUE}[1/2] Transferring root-level files...${NC}"
    rsync -avP --append-verify --timeout=60 --exclude='*/' "root@$HOST:$MOST_RECENT/" "$LOCAL_DIR/$DIR_NAME/" || true
    
    if [ -z "$REMOTE_SUBDIRS" ]; then
        echo -e "${GREEN}✓ No subdirectories to transfer${NC}"
        echo -e "${GREEN}✓ Successfully pulled $DATASET_TYPE/$DIR_NAME from $HOST${NC}"
        exit 0
    fi
    
    SUBDIRS=($REMOTE_SUBDIRS)
    TOTAL_SUBDIRS=${#SUBDIRS[@]}
    
    echo -e "${BLUE}[2/2] Transferring $TOTAL_SUBDIRS subdirectories (up to $PARALLEL at a time)...${NC}"
    
    # Function to transfer a single subdirectory with retry
    transfer_subdir() {
        local subdir=$1
        local prefix="[$subdir]"
        
        START_TIME=$(date +%s)
        TIMEOUT=300
        ATTEMPT=0
        
        # Create local subdir
        mkdir -p "$LOCAL_DIR/$DIR_NAME/$subdir"
        
        while true; do
            ATTEMPT=$((ATTEMPT + 1))
            ELAPSED=$(($(date +%s) - START_TIME))
            
            if [ $ELAPSED -ge $TIMEOUT ]; then
                echo "$prefix ✗ Failed after 5 minutes"
                return 1
            fi
            
            echo "$prefix Attempt $ATTEMPT (${ELAPSED}s elapsed)"
            
            rsync -avP --append-verify --timeout=60 \
                "root@$HOST:$MOST_RECENT/$subdir/" \
                "$LOCAL_DIR/$DIR_NAME/$subdir/" 2>&1 | \
                sed "s|^|$prefix |"
            
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "$prefix ✓ Complete"
                return 0
            fi
            
            echo "$prefix ✗ Failed, retrying..."
        done
    }
    
    export -f transfer_subdir
    export HOST MOST_RECENT LOCAL_DIR DIR_NAME
    
    # Transfer subdirectories in parallel
    FAILED=0
    for subdir in "${SUBDIRS[@]}"; do
        transfer_subdir "$subdir" &
        
        # Limit parallel jobs
        while [ $(jobs -r | wc -l) -ge $PARALLEL ]; do
            wait -n
            if [ $? -ne 0 ]; then
                FAILED=$((FAILED + 1))
            fi
        done
    done
    
    # Wait for remaining jobs
    wait
    
    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}✗ $FAILED subdirectories failed to transfer${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Successfully pulled $DATASET_TYPE/$DIR_NAME from $HOST${NC}"
    echo -e "${GREEN}  Location: $LOCAL_DIR/$DIR_NAME${NC}"
fi
