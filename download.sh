#!/bin/bash

# Array of hosts (top 4 from config)
hosts=("left" "mid" "right" "riight" "left2" "mid2" "right2" "riight2" "left3" "mid3")

# Check if arguments were provided
if [ $# -eq 0 ]; then
    # No arguments - use all hosts
    indices=$(seq 1 ${#hosts[@]})
else
    # Use provided indices
    indices=("$@")
fi

# Loop through specified indices
for i in $indices; do
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
    
    # Execute the plotting command on the remote machine
    ssh $port_option "$hostname" "cd /root/MarkovianTraining && python src/plot_training_metrics.py --window_size 100"
    
    # Find the most recently edited pg_norm_plot.png file in the results directory and its subdirectories
    latest_file=$(ssh $port_option "$hostname" "find /root/MarkovianTraining/results -name 'log_metrics.png' -print0 | xargs -0 ls -t | head -1")
    latest_log_file=$(ssh $port_option "$hostname" "find /root/MarkovianTraining/results -name 'log.jsonl' -print0 | xargs -0 ls -t | head -1")

    # Download the most recent pg_norm_plot.png file
    scp $port_option "$hostname:$latest_file" "./results_${i}_${hostname}/"
    #scp $port_option "$hostname:$latest_log_file" "./results_${i}_${hostname}/"
done
