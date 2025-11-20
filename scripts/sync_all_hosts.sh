#!/bin/bash

# Script to git pull on all hosts in SSH config (in parallel)
# - Skips hosts not on main branch
# - Runs git pull and shows any errors (git handles conflicts)
# Usage: ./scripts/sync_all_hosts.sh [parallel_jobs]
# Example: ./scripts/sync_all_hosts.sh 4

usage() {
    echo "Usage: ./scripts/sync_all_hosts.sh [parallel_jobs]"
    echo ""
    echo "Runs git pull on every host defined in ~/.ssh/config (defaults to 4 parallel jobs)."
    echo "Provide an optional integer to override the level of concurrency."
    echo "Use -h or --help to show this message."
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

PARALLEL_JOBS=${1:-4}  # Default to 4 parallel jobs
CONFIG_FILE="$HOME/.ssh/config"
REPO_DIR="~/MarkovianTraining"  # Tilde expands to remote user's home (root)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Syncing MarkovianTraining on all hosts ===${NC}"
echo -e "${BLUE}Parallel jobs: $PARALLEL_JOBS${NC}\n"

# Extract host names from SSH config (lines starting with "Host " but not control settings)
hosts=$(grep "^Host " "$CONFIG_FILE" | awk '{print $2}' | grep -v "ControlPersist")
total_hosts=$(echo "$hosts" | wc -l)

# Create temp directory for job tracking
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Function to sync a single host
sync_host() {
    local host=$1
    local job_num=$2
    local log_file="$TEMP_DIR/sync_${host}.log"
    local prefix="[$host]"
    
    echo "$prefix Connecting..."
    
    {
        echo "HOST: $host"
        
        # Check if host is reachable (with 5 second timeout)
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo 2>&1" > /dev/null 2>&1; then
            echo "STATUS: UNREACHABLE"
            echo "Cannot connect (timeout or connection refused)"
            echo "$prefix Cannot connect (timeout or connection refused)"
            exit 1
        fi
        
        # Check if repo directory exists
        if ! ssh "$host" 'test -d ~/MarkovianTraining/.git' 2>/dev/null; then
            echo "STATUS: NO_REPO"
            echo "~/MarkovianTraining does not exist or is not a git repo"
            echo "$prefix No git repository found"
            exit 1
        fi
        
        # Check current branch
        current_branch=$(ssh "$host" 'cd ~/MarkovianTraining && git branch --show-current' 2>/dev/null)
        echo "BRANCH: $current_branch"
        echo "$prefix Branch: $current_branch"
        
        if [ "$current_branch" != "main" ]; then
            echo "STATUS: WRONG_BRANCH"
            echo "Not on main branch (on $current_branch), skipping"
            echo "$prefix ⚠ Not on main branch, skipping"
            exit 0
        fi
        
        echo "$prefix Pulling..."
        
        # Pull changes - let git handle any conflicts
        pull_output=$(ssh "$host" 'cd ~/MarkovianTraining && git pull' 2>&1)
        pull_exit_code=$?
        
        if [ $pull_exit_code -eq 0 ]; then
            # Success
            if echo "$pull_output" | grep -q "Already up to date"; then
                echo "STATUS: UP_TO_DATE"
                echo "$prefix ✓ Already up to date"
            elif echo "$pull_output" | grep -q "Updating\|Fast-forward"; then
                echo "STATUS: UPDATED"
                echo "DETAILS:"
                echo "$pull_output" | grep -E "file.*changed|insertion|deletion"
                echo "$prefix ✓ Updated successfully"
            else
                # Some other success message
                echo "STATUS: UP_TO_DATE"
                echo "$prefix ✓ $pull_output"
            fi
        else
            # Git pull failed - show the actual error
            echo "STATUS: ERROR"
            echo "DETAILS:"
            echo "$pull_output"
            echo "$prefix ✗ Git pull failed"
            exit 1
        fi
        
        exit 0
    } > "$log_file" 2>&1 &
    
    # Show output in real-time (everything that's not STATUS/BRANCH/HOST/DETAILS)
    local log_pid=$!
    sleep 0.5  # Give the background process time to start
    tail -f "$log_file" 2>/dev/null | grep -v "^STATUS:\|^BRANCH:\|^HOST:\|^DETAILS:" &
    local tail_pid=$!
    
    wait $log_pid
    local result=$?
    kill $tail_pid 2>/dev/null
    wait $tail_pid 2>/dev/null
    
    return $result
}

export -f sync_host
export TEMP_DIR RED GREEN YELLOW BLUE NC

# Launch jobs in parallel
job_num=0
for host in $hosts; do
    job_num=$((job_num + 1))
    sync_host "$host" "$job_num" &
    
    # Limit parallel jobs
    if [ $(jobs -r | wc -l) -ge $PARALLEL_JOBS ]; then
        wait -n
    fi
done

# Wait for all jobs to complete
wait

echo ""
echo -e "${BLUE}=== Summary ===${NC}\n"

job_num=0
up_to_date=0
updated=0
skipped=0
errors=0

for host in $hosts; do
    job_num=$((job_num + 1))
    log_file="$TEMP_DIR/sync_${host}.log"
    
    if [ ! -f "$log_file" ]; then
        echo -e "${RED}[$job_num/$total_hosts] $host: No log found${NC}"
        errors=$((errors + 1))
        continue
    fi
    
    status=$(grep "^STATUS:" "$log_file" | cut -d' ' -f2)
    branch=$(grep "^BRANCH:" "$log_file" | cut -d' ' -f2-)
    
    case "$status" in
        UNREACHABLE)
            echo -e "${RED}[$job_num/$total_hosts] $host: Cannot connect${NC}"
            msg=$(grep -v "^HOST:\|^STATUS:\|^BRANCH:" "$log_file")
            [ -n "$msg" ] && echo -e "${RED}  $msg${NC}"
            errors=$((errors + 1))
            ;;
        NO_REPO)
            echo -e "${YELLOW}[$job_num/$total_hosts] $host: No git repository${NC}"
            skipped=$((skipped + 1))
            ;;
        WRONG_BRANCH)
            echo -e "${YELLOW}[$job_num/$total_hosts] $host: On branch '$branch' (not main), skipping${NC}"
            skipped=$((skipped + 1))
            ;;
        UP_TO_DATE)
            echo -e "${GREEN}[$job_num/$total_hosts] $host: Already up to date (branch: $branch)${NC}"
            up_to_date=$((up_to_date + 1))
            ;;
        UPDATED)
            echo -e "${GREEN}[$job_num/$total_hosts] $host: Successfully updated (branch: $branch)${NC}"
            details=$(sed -n '/^DETAILS:/,$ p' "$log_file" | grep -v "^DETAILS:")
            [ -n "$details" ] && echo "$details" | sed 's/^/  /'
            updated=$((updated + 1))
            ;;
        ERROR)
            echo -e "${RED}[$job_num/$total_hosts] $host: Pull error (branch: $branch)${NC}"
            details=$(sed -n '/^DETAILS:/,$ p' "$log_file" | grep -v "^DETAILS:")
            [ -n "$details" ] && echo "$details" | head -5 | sed 's/^/  /'
            errors=$((errors + 1))
            ;;
        *)
            echo -e "${RED}[$job_num/$total_hosts] $host: Unknown status${NC}"
            errors=$((errors + 1))
            ;;
    esac
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary:${NC}"
echo -e "  Total hosts: $total_hosts"
echo -e "  ${GREEN}Updated: $updated${NC}"
echo -e "  ${GREEN}Already up to date: $up_to_date${NC}"
echo -e "  ${YELLOW}Skipped: $skipped${NC}"
echo -e "  ${RED}Errors: $errors${NC}"
echo -e "${BLUE}========================================${NC}"

if [ $errors -gt 0 ]; then
    exit 1
fi
