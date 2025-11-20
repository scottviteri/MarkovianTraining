#!/bin/bash

# Push results from local ICLR2026Rebuttals/Logs folder to remote host
# Usage: ./scripts/push_results.sh --source <host:dataset[:timestamp_or_index]> --target <host> [--parallel N] [--all]
# Examples:
#   ./scripts/push_results.sh --source mid:aqua --target right3
#   ./scripts/push_results.sh --source mid:aqua --target right3 --parallel 4
#   ./scripts/push_results.sh --source mid:aqua --target right3 --all --parallel 4  # Push all (4 at once)
#   ./scripts/push_results.sh --source mid:aqua:-1 --target right3
#   ./scripts/push_results.sh --source mid:aqua:20251116_193803 --target right3

# Note: Don't use 'set -e' because we want to handle rsync failures and retry

usage() {
    echo "Usage: $0 --source <host:dataset[:timestamp_or_index]> --target <host> [--parallel N] [--all]"
    echo ""
    echo "Examples:"
    echo "  $0 --source mid:aqua --target right3"
    echo "  $0 --source mid:aqua --target right3 --parallel 4"
    echo "  $0 --source mid:aqua --target right3 --all              # Push all aqua runs"
    echo "  $0 --source mid:aqua --target right3 --all --parallel 4 # Push all (4 at a time)"
    echo "  $0 --source mid:aqua:-1 --target right3                 # 2nd most recent"
    echo "  $0 --source mid:aqua:20251116_193803 --target right3"
    echo ""
    echo "Options:"
    echo "  --parallel N    Transfer N subdirectories simultaneously (single run)"
    echo "                  or N runs simultaneously (with --all) (default: 1)"
    echo "  --all           Push all runs matching the spec (ignores timestamp/index)"
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
TARGET=""
PARALLEL=1
PUSH_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --all)
            PUSH_ALL=true
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

if [ -z "$SOURCE" ] || [ -z "$TARGET" ]; then
    echo "Error: Both --source and --target are required"
    usage
fi

# Parse source (format: host:dataset[:timestamp_or_index])
IFS=':' read -r SOURCE_HOST DATASET TIMESTAMP_OR_INDEX <<< "$SOURCE"

if [ -z "$SOURCE_HOST" ] || [ -z "$DATASET" ]; then
    echo "Error: Source must be in format host:dataset[:timestamp_or_index]"
    usage
fi

DEST_HOST="$TARGET"
LOCAL_BASE="/home/scottviteri/Projects/MarkovianTraining/ICLR2026Rebuttals/Logs"
REMOTE_BASE="/root/MarkovianTraining/results"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Handle --all flag
if [ "$PUSH_ALL" = true ]; then
    if [ -z "$DATASET" ]; then
        echo -e "${RED}Error: --all requires a dataset to be specified${NC}"
        echo "Usage: $0 --source host:dataset --target desthost --all"
        exit 1
    fi
    
    if [ -n "$TIMESTAMP_OR_INDEX" ]; then
        echo -e "${YELLOW}Warning: --all ignores timestamp/index${NC}"
    fi
    
    LOCAL_DATASET_DIR="$LOCAL_BASE/$SOURCE_HOST/$DATASET"
    
    if [ ! -d "$LOCAL_DATASET_DIR" ]; then
        echo -e "${RED}Directory not found: $LOCAL_DATASET_DIR${NC}"
        echo -e "${RED}Make sure you have pulled data from $SOURCE_HOST for dataset $DATASET${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Finding all $DATASET runs from $SOURCE_HOST${NC}"
    
    # Get all local directories sorted by modification time
    ALL_DIRS=$(ls -t "$LOCAL_DATASET_DIR" 2>/dev/null)
    
    if [ -z "$ALL_DIRS" ]; then
        echo -e "${RED}No $DATASET runs found in $LOCAL_DATASET_DIR${NC}"
        exit 1
    fi
    
    DIR_COUNT=$(echo "$ALL_DIRS" | wc -l)
    echo -e "${GREEN}Found $DIR_COUNT runs to push${NC}"
    echo -e "${YELLOW}Parallelism: ${NC}$PARALLEL"
    echo ""
    
    # Create temp directory for tracking
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    # Function to push a single run
    push_single_run() {
        local DIR_NAME=$1
        local RUN_NUM=$2
        local PREFIX="[$RUN_NUM/$DIR_COUNT: $DIR_NAME]"
        local STATUS_FILE="$TEMP_DIR/status_$RUN_NUM"
        
        echo "$PREFIX Starting..."
        
        LOCAL_DIR="$LOCAL_DATASET_DIR/$DIR_NAME"
        REMOTE_PATH="$REMOTE_BASE/$DATASET"
        
        # Create remote directory if needed
        ssh "$DEST_HOST" "mkdir -p $REMOTE_PATH/$DIR_NAME" 2>/dev/null
        
        # Push with single attempt (no retry loop in --all mode for speed)
        rsync -avP --append-verify --timeout=60 \
            "$LOCAL_DIR/" \
            "root@$DEST_HOST:$REMOTE_PATH/$DIR_NAME/" 2>&1 | \
            sed "s|^|$PREFIX |"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "$PREFIX ✓ Complete"
            echo "SUCCESS" > "$STATUS_FILE"
            return 0
        else
            echo "$PREFIX ✗ Failed"
            echo "FAILED" > "$STATUS_FILE"
            return 1
        fi
    }
    
    export -f push_single_run
    export SOURCE_HOST DATASET DEST_HOST LOCAL_DATASET_DIR REMOTE_BASE TEMP_DIR DIR_COUNT
    
    # Push directories in parallel
    CURRENT=0
    for DIR_NAME in $ALL_DIRS; do
        CURRENT=$((CURRENT + 1))
        push_single_run "$DIR_NAME" "$CURRENT" &
        
        # Limit parallel jobs
        while [ $(jobs -r | wc -l) -ge $PARALLEL ]; do
            wait -n
        done
    done
    
    # Wait for all remaining jobs
    wait
    
    # Count results
    SUCCESS=0
    FAILED=0
    for i in $(seq 1 $DIR_COUNT); do
        STATUS_FILE="$TEMP_DIR/status_$i"
        if [ -f "$STATUS_FILE" ]; then
            if grep -q "SUCCESS" "$STATUS_FILE"; then
                SUCCESS=$((SUCCESS + 1))
            else
                FAILED=$((FAILED + 1))
            fi
        else
            FAILED=$((FAILED + 1))
        fi
    done
    
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

# Build local path for this source host and dataset
LOCAL_DATASET_DIR="$LOCAL_BASE/$SOURCE_HOST/$DATASET"

if [ ! -d "$LOCAL_DATASET_DIR" ]; then
    echo -e "${RED}Directory not found: $LOCAL_DATASET_DIR${NC}"
    echo -e "${RED}Make sure you have pulled data from $SOURCE_HOST for dataset $DATASET${NC}"
    exit 1
fi

# Find the directory to push based on timestamp or index
if [[ "$TIMESTAMP_OR_INDEX" =~ ^-[0-9]+$ ]]; then
    # Negative index provided
    INDEX=${TIMESTAMP_OR_INDEX#-}  # Remove the minus sign
    INDEX=$((INDEX + 1))  # Convert: -1 becomes 2nd, -2 becomes 3rd, etc.
    
    echo -e "${BLUE}Finding ${INDEX}th most recent $DATASET results from $SOURCE_HOST${NC}"
    
    # Get list of directories sorted by recency, pick the Nth one
    TIMESTAMP=$(ls -t "$LOCAL_DATASET_DIR" 2>/dev/null | sed -n "${INDEX}p")
    
    if [ -z "$TIMESTAMP" ]; then
        TOTAL=$(ls -t "$LOCAL_DATASET_DIR" 2>/dev/null | wc -l)
        echo -e "${RED}No directory found at index $TIMESTAMP_OR_INDEX (only $TOTAL directories available)${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Found: $TIMESTAMP${NC}"
    
elif [ -n "$TIMESTAMP_OR_INDEX" ]; then
    # Specific timestamp provided
    TIMESTAMP="$TIMESTAMP_OR_INDEX"
    echo -e "${BLUE}Using specific timestamp: $TIMESTAMP${NC}"
    
else
    # Find most recent directory (default)
    echo -e "${BLUE}Finding most recent $DATASET results from $SOURCE_HOST${NC}"
    TIMESTAMP=$(ls -t "$LOCAL_DATASET_DIR" 2>/dev/null | head -1)
    
    if [ -z "$TIMESTAMP" ]; then
        echo -e "${RED}No directories found in $LOCAL_DATASET_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Found most recent: $TIMESTAMP${NC}"
fi

LOCAL_DIR="$LOCAL_DATASET_DIR/$TIMESTAMP"

if [ ! -d "$LOCAL_DIR" ]; then
    echo -e "${RED}Directory not found: $LOCAL_DIR${NC}"
    exit 1
fi

REMOTE_PATH="$REMOTE_BASE/$DATASET"

echo ""
echo -e "${YELLOW}Source: ${NC}$SOURCE_HOST/$DATASET/$TIMESTAMP"
echo -e "${YELLOW}Target: ${NC}$DEST_HOST:$REMOTE_PATH/$TIMESTAMP"
echo -e "${YELLOW}Parallelism: ${NC}$PARALLEL"
echo ""

# Create remote directory if needed
ssh "$DEST_HOST" "mkdir -p $REMOTE_PATH/$TIMESTAMP" 2>/dev/null

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
            echo -e "${RED}✗ Failed to push results after 5 minutes${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}Attempt $ATTEMPT (${ELAPSED}s elapsed)${NC}"
        
        rsync -avP --append-verify --timeout=60 "$LOCAL_DIR/" "root@$DEST_HOST:$REMOTE_PATH/$TIMESTAMP/"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Successfully pushed $TIMESTAMP to $DEST_HOST:$REMOTE_PATH/${NC}"
            exit 0
        fi
        
        echo -e "${RED}✗ Transfer failed, retrying immediately...${NC}"
    done
else
    # Parallel mode - transfer subdirectories in parallel
    echo -e "${BLUE}Transferring (parallel mode)${NC}"
    
    # First, transfer root-level files (non-recursive)
    echo -e "${BLUE}[1/2] Transferring root-level files...${NC}"
    rsync -avP --append-verify --timeout=60 --exclude='*/' "$LOCAL_DIR/" "root@$DEST_HOST:$REMOTE_PATH/$TIMESTAMP/" || true
    
    # Get list of subdirectories
    SUBDIRS=($(find "$LOCAL_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))
    TOTAL_SUBDIRS=${#SUBDIRS[@]}
    
    if [ $TOTAL_SUBDIRS -eq 0 ]; then
        echo -e "${GREEN}✓ No subdirectories to transfer${NC}"
        exit 0
    fi
    
    echo -e "${BLUE}[2/2] Transferring $TOTAL_SUBDIRS subdirectories (up to $PARALLEL at a time)...${NC}"
    
    # Function to transfer a single subdirectory with retry
    transfer_subdir() {
        local subdir=$1
        local prefix="[$subdir]"
        
        START_TIME=$(date +%s)
        TIMEOUT=300
        ATTEMPT=0
        
        while true; do
            ATTEMPT=$((ATTEMPT + 1))
            ELAPSED=$(($(date +%s) - START_TIME))
            
            if [ $ELAPSED -ge $TIMEOUT ]; then
                echo "$prefix ✗ Failed after 5 minutes"
                return 1
            fi
            
            echo "$prefix Attempt $ATTEMPT (${ELAPSED}s elapsed)"
            
            rsync -avP --append-verify --timeout=60 \
                "$LOCAL_DIR/$subdir/" \
                "root@$DEST_HOST:$REMOTE_PATH/$TIMESTAMP/$subdir/" 2>&1 | \
                sed "s|^|$prefix |"
            
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "$prefix ✓ Complete"
                return 0
            fi
            
            echo "$prefix ✗ Failed, retrying..."
        done
    }
    
    export -f transfer_subdir
    export DEST_HOST REMOTE_PATH TIMESTAMP LOCAL_DIR
    
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
    
    echo -e "${GREEN}✓ Successfully pushed $TIMESTAMP to $DEST_HOST:$REMOTE_PATH/${NC}"
fi
