#!/bin/bash

# Array of hosts (top 4 from config)
hosts=("Wiki" "LlamaCtxt200" "Llama30add" "4mst")

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
    
    # Download the specific file with a unique name
    scp $port_option "$hostname:/root/MarkovianTraining/src/AnalyzeResults/pg_norm_plot.png" "./results_${i}_${hostname}_pg_norm_plot.png"
done