#!/bin/bash

# Array of hosts
hosts=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13")

# Parse options
COPY_FROM=""
COPY_TO=""
PULL=false
SCP=false
INDICES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --copy_from_to)
            COPY_FROM="$2"
            COPY_TO="$3"
            shift 3
            ;;
        --pull)
            PULL=true
            shift
            ;;
        --scp)
            SCP=true
            shift
            ;;
        *)
            INDICES+=("$1")
            shift
            ;;
    esac
done

# If no indices provided and not copying, use all hosts
if [ ${#INDICES[@]} -eq 0 ] && [ -z "$COPY_FROM" ]; then
    INDICES=($(seq 1 ${#hosts[@]}))
fi

# Handle copy operation if requested
if [ ! -z "$COPY_FROM" ] && [ ! -z "$COPY_TO" ]; then
    # Get the last element as destination
    dst_idx=${COPY_TO}
    
    # Validate destination index
    if [ "$dst_idx" -lt 1 ] || [ "$dst_idx" -gt ${#hosts[@]} ]; then
        echo "Invalid destination index: $dst_idx (must be between 1 and ${#hosts[@]})"
        exit 1
    fi
    
    # Get destination host
    dst_host="${hosts[$((dst_idx-1))]}"
    IFS=':' read -r dst_hostname dst_port <<< "$dst_host"
    dst_port_option=""
    if [ ! -z "$dst_port" ]; then
        dst_port_option="-P $dst_port"
    fi
    
    # Process each source index
    for src_idx in ${COPY_FROM}; do
        # Validate source index
        if [ "$src_idx" -lt 1 ] || [ "$src_idx" -gt ${#hosts[@]} ]; then
            echo "Invalid source index: $src_idx (must be between 1 and ${#hosts[@]})"
            continue
        fi
        
        # Get source host
        src_host="${hosts[$((src_idx-1))]}"
        IFS=':' read -r src_hostname src_port <<< "$src_host"
        src_port_option=""
        if [ ! -z "$src_port" ]; then
            src_port_option="-P $src_port"
        fi
        
        # Get the source directory and log file
        src_dir="results_${src_idx}_${src_hostname}"
        src_log="${src_dir}/log.jsonl"
        
        if [ ! -f "$src_log" ]; then
            echo "Source log file not found: $src_log"
            continue
        fi
        
        # Create the same directory structure on target machine
        echo "Creating directory on target machine for source $src_idx..."
        ssh $dst_port_option "$dst_hostname" "cd /root/MarkovianTraining && mkdir -p $src_dir"
        
        # Copy to destination with same directory structure
        echo "Copying log from host $src_hostname to $dst_hostname..."
        scp $dst_port_option "$src_log" "$dst_hostname:/root/MarkovianTraining/${src_dir}/log.jsonl"
    done
    exit 0
fi

# Process each specified index
for i in "${INDICES[@]}"; do
    # Validate index
    if [ "$i" -lt 1 ] || [ "$i" -gt ${#hosts[@]} ]; then
        echo "Invalid index: $i (must be between 1 and ${#hosts[@]})"
        continue
    fi
    
    # Get host (subtract 1 as array is 0-based)
    host="${hosts[$((i-1))]}"
    
    # Split host and port if specified
    IFS=':' read -r hostname port <<< "$host"
    
    # If no port specified, use the one from ssh config
    port_option=""
    if [ ! -z "$port" ]; then
        port_option="-P $port"
    fi
    
    # Create a unique directory for each host
    mkdir -p "./results_${i}_${hostname}"
    
    # Execute git pull if requested
    if [ "$PULL" = true ]; then
        echo "Running git pull on host $hostname..."
        ssh $port_option "$hostname" "cd /root/MarkovianTraining && git pull"
    fi
    
    # Execute scp commands if requested
    if [ "$SCP" = true ]; then
        echo "Downloading files from host $hostname..."
        
        # Find the most recently edited files
        latest_file=$(ssh $port_option "$hostname" "find /root/MarkovianTraining/results -name 'log_metrics.png' -print0 | xargs -0 ls -t | head -1")
        latest_log_file=$(ssh $port_option "$hostname" "find /root/MarkovianTraining/results -name 'log.jsonl' -print0 | xargs -0 ls -t | head -1")

        # Download the files
        scp $port_option "$hostname:$latest_file" "./results_${i}_${hostname}/"
        scp $port_option "$hostname:$latest_log_file" "./results_${i}_${hostname}/"
    fi
    
    # If neither operation is requested, show usage
    if [ "$PULL" = false ] && [ "$SCP" = false ] && [ -z "$COPY_FROM" ]; then
        echo "No operation specified. Use --pull to run git pull or --scp to download files."
        echo "Usage:"
        echo "  ./download.sh [indices] --pull    # Run git pull on specified machines"
        echo "  ./download.sh [indices] --scp     # Download files from specified machines"
        echo "  ./download.sh --copy_from_to i j  # Copy files from machine i to machine j"
        exit 1
    fi
done
