#!/bin/bash

# Script to collect ONLY the most recent experiment from each machine
# Automatically removes previous experiments from the same machine
#
# Usage:
#   ./collect_remote_experiments.sh                    # Collect from all machines
#   ./collect_remote_experiments.sh 1                  # Collect from tier 1: left, mid, right, riight
#   ./collect_remote_experiments.sh 2                  # Collect from tier 2: left2, mid2, right2, riight2  
#   ./collect_remote_experiments.sh 3                  # Collect from tier 3: left3, mid3, right3, riight3
#   ./collect_remote_experiments.sh 1 left             # Collect only from left (tier 1)
#   ./collect_remote_experiments.sh 2 mid2             # Collect only from mid2 (tier 2)

set -e

# Define machines from SSH config (with their user paths)
declare -A MACHINES=(
    ["mid3"]="/root/MarkovianTraining"
    ["left3"]="/root/MarkovianTraining" 
    ["riight3"]="/home/ubuntu/MarkovianTraining"
    ["right3"]="/root/MarkovianTraining"
    ["riight2"]="/home/ubuntu/MarkovianTraining"
    ["right2"]="/root/MarkovianTraining"
    ["mid2"]="/home/ubuntu/MarkovianTraining"
    ["left2"]="/root/MarkovianTraining"
    ["riight"]="/root/MarkovianTraining"
    ["right"]="/root/MarkovianTraining"
    ["mid"]="/home/ubuntu/MarkovianTraining"
    ["left"]="/home/ubuntu/MarkovianTraining"
)

# Parse arguments
tier="$1"
specific_machine="$2"

# Function to get machines for a specific tier
get_tier_machines() {
    local tier_num="$1"
    case "$tier_num" in
        "1")
            echo "left mid right riight"
            ;;
        "2") 
            echo "left2 mid2 right2 riight2"
            ;;
        "3")
            echo "left3 mid3 right3 riight3"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Determine which machines to collect from
if [ -n "$specific_machine" ]; then
    # Specific machine specified
    if [[ ! " ${!MACHINES[@]} " =~ " ${specific_machine} " ]]; then
        echo "❌ Error: Machine '$specific_machine' not found in SSH config"
        echo "Available machines: ${!MACHINES[@]}"
        exit 1
    fi
    target_machines=("$specific_machine")
    echo "🎯 Collecting from specific machine: $specific_machine"
elif [ -n "$tier" ]; then
    # Tier specified
    tier_machines=$(get_tier_machines "$tier")
    if [ -z "$tier_machines" ]; then
        echo "❌ Error: Invalid tier '$tier'. Valid tiers: 1, 2, 3"
        exit 1
    fi
    read -ra target_machines <<< "$tier_machines"
    echo "🎯 Collecting from tier $tier machines: ${target_machines[*]}"
else
    # No arguments - collect from all machines
    target_machines=($(printf "%s\n" "${!MACHINES[@]}" | sort))
    echo "🚀 Collecting from ALL machines: ${target_machines[*]}"
fi

echo "==============================================================="

# Function to remove previous experiments from the same machine
cleanup_previous_experiments() {
    local machine="$1"
    local task_type="$2"
    local new_timestamp="$3"
    
    local task_dir="results/$task_type"
    
    if [ ! -d "$task_dir" ]; then
        return
    fi
    
    # Find and remove previous experiments from this machine
    find "$task_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${machine}" | while read exp_dir; do
        exp_name=$(basename "$exp_dir")
        exp_timestamp=$(echo "$exp_name" | cut -d'_' -f1)
        
        # Remove if it's not the new experiment we're about to add
        if [ "$exp_timestamp" != "$new_timestamp" ]; then
            echo "      🗑️  Removing previous experiment: $exp_name"
            rm -rf "$exp_dir"
        fi
    done
}

# Function to process a log file and place it correctly
place_experiment() {
    local source_file="$1"
    local machine="$2"
    local original_timestamp="$3"
    
    if [ ! -f "$source_file" ]; then
        echo "      ❌ Source file not found: $source_file"
        return
    fi
    
    # Get task type from the log file
    first_line=$(head -n 1 "$source_file" 2>/dev/null || echo "")
    task_type=$(echo "$first_line" | jq -r '.task_type // empty' 2>/dev/null || echo "")
    
    if [ -z "$task_type" ]; then
        echo "      ⚠️  Could not determine task type for $(basename "$source_file")"
        return
    fi
    
    # Clean up previous experiments from this machine for this task type
    cleanup_previous_experiments "$machine" "$task_type" "$original_timestamp"
    
    # Use original timestamp but add machine suffix for uniqueness
    target_dir="results/$task_type/${original_timestamp}_${machine}"
    mkdir -p "$target_dir"
    
    # Copy the log file
    cp "$source_file" "$target_dir/log.jsonl"
    
    # Get log stats
    lines=$(wc -l < "$target_dir/log.jsonl" 2>/dev/null || echo 0)
    
    echo "      ✅ $task_type: ${original_timestamp}_${machine} ($lines lines)"
}

# Collect from each target machine
for machine in "${target_machines[@]}"; do
    if [[ ! " ${!MACHINES[@]} " =~ " ${machine} " ]]; then
        echo "⚠️  Machine '$machine' not found in SSH config, skipping"
        continue
    fi
    
    base_path="${MACHINES[$machine]}"
    echo ""
    echo "📡 Connecting to $machine (path: $base_path)..."
    
    # Test connection first
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$machine" "echo 'Connection successful'" >/dev/null 2>&1; then
        echo "❌ Failed to connect to $machine (skipping)"
        continue
    fi
    
    echo "✅ Connected to $machine"
    
    # Find the single most recent experiment folder or log file
    echo "   🔍 Finding most recent experiment..."
    
    most_recent=$(ssh "$machine" "
        # Look for experiment folders first (they're usually the main experiments)
        recent_folders=\$(find '$base_path/results' -mindepth 2 -maxdepth 2 -type d -name '[0-9]*' 2>/dev/null | 
        while read dir; do
            if [ -d \"\$dir\" ] && [ -f \"\$dir/log.jsonl\" ]; then
                echo \"\$dir|\$(stat -c %Y \"\$dir\" 2>/dev/null || echo 0)\"
            fi
        done | sort -t'|' -k2 -nr | head -1 | cut -d'|' -f1)
        
        # Also look for standalone log files
        recent_files=\$(find '$base_path/results' -name '*.jsonl' -not -path '*/Official/*' 2>/dev/null | 
        while read file; do
            if [ -f \"\$file\" ]; then
                echo \"\$file|\$(stat -c %Y \"\$file\" 2>/dev/null || echo 0)\"
            fi
        done | sort -t'|' -k2 -nr | head -1 | cut -d'|' -f1)
        
        # Get modification times to compare
        folder_time=0
        file_time=0
        
        if [ -n \"\$recent_folders\" ]; then
            folder_time=\$(stat -c %Y \"\$recent_folders\" 2>/dev/null || echo 0)
        fi
        
        if [ -n \"\$recent_files\" ]; then
            file_time=\$(stat -c %Y \"\$recent_files\" 2>/dev/null || echo 0)
        fi
        
        # Return the most recent between folder and file
        if [ \$folder_time -gt \$file_time ]; then
            echo \"folder|\$recent_folders\"
        elif [ \$file_time -gt 0 ]; then
            echo \"file|\$recent_files\"
        fi
    " 2>/dev/null || echo "")
    
    if [ -n "$most_recent" ]; then
        IFS='|' read -r type path <<< "$most_recent"
        
        if [ "$type" = "folder" ]; then
            folder_name=$(basename "$path")
            echo "   📁 Most recent: experiment folder $folder_name"
            
            # Copy the entire folder
            temp_folder="temp_${machine}_${folder_name}"
            if scp -r -q "$machine:$path" "$temp_folder" 2>/dev/null; then
                echo "      📥 Downloaded experiment folder"
                
                # Process the log file
                if [ -f "$temp_folder/log.jsonl" ]; then
                    place_experiment "$temp_folder/log.jsonl" "$machine" "$folder_name"
                fi
                
                # Clean up temp folder
                rm -rf "$temp_folder"
            else
                echo "      ❌ Failed to download folder"
            fi
            
        elif [ "$type" = "file" ]; then
            filename=$(basename "$path")
            # Extract timestamp from filename or use current time
            timestamp=$(echo "$filename" | grep -o '[0-9]\{8\}_[0-9]\{6\}' | head -1)
            if [ -z "$timestamp" ]; then
                timestamp=$(date '+%Y%m%d_%H%M%S')
            fi
            
            echo "   📄 Most recent: log file $filename"
            
            # Copy the log file
            temp_file="temp_${machine}_${filename}"
            if scp -q "$machine:$path" "$temp_file" 2>/dev/null; then
                echo "      📥 Downloaded log file"
                
                # Process the log file
                place_experiment "$temp_file" "$machine" "$timestamp"
                
                # Clean up temp file
                rm -f "$temp_file"
            else
                echo "      ❌ Failed to download file"
            fi
        fi
    else
        echo "   ⚠️  No recent experiments found on $machine"
    fi
done

echo ""
echo "==============================================================="
echo "🎉 Collection and cleanup complete!"

echo ""
echo "📁 Updated experiment count by task type:"
for task_dir in results/*/; do
    if [ -d "$task_dir" ]; then
        task_type=$(basename "$task_dir")
        # Only count experiments with the new timestamp_machine pattern
        recent_experiments=$(find "$task_dir" -mindepth 1 -maxdepth 1 -type d -name "*_*" | wc -l)
        if [ "$recent_experiments" -gt 0 ]; then
            echo "   📊 $task_type: $recent_experiments experiments"
            
            # Show which machines were updated if we're doing selective collection
            if [ ${#target_machines[@]} -lt ${#MACHINES[@]} ]; then
                updated_machines=()
                for machine in "${target_machines[@]}"; do
                    if find "$task_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${machine}" | grep -q .; then
                        updated_machines+=("$machine")
                    fi
                done
                if [ ${#updated_machines[@]} -gt 0 ]; then
                    echo "      Updated machines: ${updated_machines[*]}"
                fi
            fi
        fi
    fi
done

echo ""
echo "🔬 Ready for analysis!"
echo "   python src/plot_training_metrics.py -f results/wiki_continuation/*/log.jsonl"

# Show usage examples
echo ""
echo "💡 Usage examples:"
echo "   ./collect_remote_experiments.sh           # Collect from all machines"
echo "   ./collect_remote_experiments.sh 1         # Collect from tier 1 (left, mid, right, riight)"
echo "   ./collect_remote_experiments.sh 2         # Collect from tier 2 (left2, mid2, right2, riight2)"
echo "   ./collect_remote_experiments.sh 1 left    # Collect only from left machine"