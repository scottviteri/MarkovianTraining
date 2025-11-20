#!/bin/bash

# Script to collect ONLY the most recent experiment from each machine
# Automatically removes previous experiments from the same machine
#
# Usage:
#   ./scripts/collect_remote_experiments.sh                    # Collect from all machines
#   ./scripts/collect_remote_experiments.sh 1                  # Collect from tier 1: left, mid, right, riight
#   ./scripts/collect_remote_experiments.sh 2                  # Collect from tier 2: left2, mid2, right2, riight2  
#   ./scripts/collect_remote_experiments.sh 3                  # Collect from tier 3: left3, mid3, right3, riight3
#   ./scripts/collect_remote_experiments.sh 1 left             # Collect only from left (tier 1)
#   ./scripts/collect_remote_experiments.sh 2 mid2             # Collect only from mid2 (tier 2)
#
# Optional flags:
#   --adapters                                          # Also collect adapter directories (not just log files)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

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
collect_adapters=false
tier=""
specific_machine=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Script to collect ONLY the most recent experiment from each machine"
            echo "Automatically removes previous experiments from the same machine"
            echo ""
            echo "Usage:"
            echo "  ./scripts/collect_remote_experiments.sh                    # Collect from all machines"
            echo "  ./scripts/collect_remote_experiments.sh 1                  # Collect from tier 1: left, mid, right, riight"
            echo "  ./scripts/collect_remote_experiments.sh 2                  # Collect from tier 2: left2, mid2, right2, riight2"
            echo "  ./scripts/collect_remote_experiments.sh 3                  # Collect from tier 3: left3, mid3, right3, riight3"
            echo "  ./scripts/collect_remote_experiments.sh 1 left             # Collect only from left (tier 1)"
            echo "  ./scripts/collect_remote_experiments.sh 2 mid2             # Collect only from mid2 (tier 2)"
            echo ""
            echo "Optional flags:"
            echo "  --adapters                                          # Also collect adapter directories (not just log files)"
            echo "  --help, -h                                          # Show this help message"
            exit 0
            ;;
        --adapters)
            collect_adapters=true
            shift
            ;;
        [1-3])
            tier="$1"
            shift
            ;;
        *)
            # If it's not a flag and we don't have a tier yet, it's the tier
            if [ -z "$tier" ] && [[ "$1" =~ ^[1-3]$ ]]; then
                tier="$1"
            # If we have a tier but no specific machine, it's the machine
            elif [ -n "$tier" ] && [ -z "$specific_machine" ]; then
                specific_machine="$1"
            # If no tier is specified, the first non-flag argument is either tier or machine
            elif [ -z "$tier" ] && [ -z "$specific_machine" ]; then
                # Check if it's a valid tier number
                if [[ "$1" =~ ^[1-3]$ ]]; then
                    tier="$1"
                else
                    # Assume it's a specific machine name
                    specific_machine="$1"
                fi
            fi
            shift
            ;;
    esac
done

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

# Show adapter collection status
if [ "$collect_adapters" = true ]; then
    echo "📦 Adapter collection: ENABLED"
else
    echo "📦 Adapter collection: DISABLED (use --adapters to enable)"
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
    local source_folder="$4"  # Optional: source folder for adapters
    local remote_path="$5"    # Full remote path to extract task type from
    
    if [ ! -f "$source_file" ]; then
        echo "      ❌ Source file not found: $source_file"
        return
    fi
    
    # Get task type from the remote folder path structure
    # Expected structure: /path/to/MarkovianTraining/results/task_type/timestamp_folder
    task_type=""
    if [ -n "$remote_path" ]; then
        # Extract task type from path: get the parent directory name of the experiment folder
        task_type=$(echo "$remote_path" | sed 's|.*/results/\([^/]*\)/.*|\1|')
    fi
    
    # Fallback: try to get task type from the log file if path extraction failed
    if [ -z "$task_type" ]; then
        first_line=$(head -n 1 "$source_file" 2>/dev/null || echo "")
        task_type=$(echo "$first_line" | jq -r '.task_type // empty' 2>/dev/null || echo "")
    fi
    
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
    
    # Copy adapters if requested and source folder is provided
    if [ "$collect_adapters" = true ] && [ -n "$source_folder" ] && [ -d "$source_folder" ]; then
        adapter_count=0
        for adapter_dir in "$source_folder"/adapter_*; do
            if [ -d "$adapter_dir" ]; then
                adapter_name=$(basename "$adapter_dir")
                echo "        📦 Copying adapter: $adapter_name"
                cp -r "$adapter_dir" "$target_dir/"
                adapter_count=$((adapter_count + 1))
            fi
        done
        if [ $adapter_count -gt 0 ]; then
            echo "        ✅ Copied $adapter_count adapter(s)"
        else
            echo "        ⚠️  No adapter directories found"
        fi
    fi
    
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
    
    # Test connection first (try with agent forwarding and without strict host checking)
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no "$machine" "echo 'Connection successful'" >/dev/null 2>&1; then
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
                # Extract timestamp from folder name (format: YYYYMMDD_HHMMSS)
                folder_name=\$(basename \"\$dir\")
                timestamp=\$(echo \"\$folder_name\" | grep -o '^[0-9]\{8\}_[0-9]\{6\}' | head -1)
                if [ -n \"\$timestamp\" ]; then
                    # Convert to sortable format and output with path
                    echo \"\$dir|\$timestamp\"
                fi
            fi
        done | sort -t'|' -k2 -r | head -1 | cut -d'|' -f1)
        
        # Also look for standalone log files
        recent_files=\$(find '$base_path/results' -name '*.jsonl' -not -path '*/Official/*' 2>/dev/null | 
        while read file; do
            if [ -f \"\$file\" ]; then
                # Extract timestamp from filename
                filename=\$(basename \"\$file\")
                timestamp=\$(echo \"\$filename\" | grep -o '[0-9]\{8\}_[0-9]\{6\}' | head -1)
                if [ -n \"\$timestamp\" ]; then
                    echo \"\$file|\$timestamp\"
                else
                    # Fallback to file modification time if no timestamp in name
                    echo \"\$file|\$(stat -c %Y \"\$file\" 2>/dev/null || echo 0)\"
                fi
            fi
        done | sort -t'|' -k2 -r | head -1 | cut -d'|' -f1)
        
        # Get timestamps to compare (folder timestamps are in YYYYMMDD_HHMMSS format)
        folder_timestamp=\"\"
        file_timestamp=\"\"
        
        if [ -n \"\$recent_folders\" ]; then
            folder_name=\$(basename \"\$recent_folders\")
            folder_timestamp=\$(echo \"\$folder_name\" | grep -o '^[0-9]\{8\}_[0-9]\{6\}' | head -1)
        fi
        
        if [ -n \"\$recent_files\" ]; then
            filename=\$(basename \"\$recent_files\")
            file_timestamp=\$(echo \"\$filename\" | grep -o '[0-9]\{8\}_[0-9]\{6\}' | head -1)
            # If no timestamp in filename, use modification time (less reliable)
            if [ -z \"\$file_timestamp\" ]; then
                file_timestamp=\$(stat -c %Y \"\$recent_files\" 2>/dev/null || echo 0)
            fi
        fi
        
        # Return the most recent between folder and file (comparing timestamps)
        if [ -n \"\$folder_timestamp\" ] && [ -n \"\$file_timestamp\" ]; then
            if [[ \"\$folder_timestamp\" > \"\$file_timestamp\" ]]; then
                echo \"folder|\$recent_folders\"
            else
                echo \"file|\$recent_files\"
            fi
        elif [ -n \"\$folder_timestamp\" ]; then
            echo \"folder|\$recent_folders\"
        elif [ -n \"\$file_timestamp\" ]; then
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
                    place_experiment "$temp_folder/log.jsonl" "$machine" "$folder_name" "$temp_folder" "$path"
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
                
                # Process the log file (no source folder for standalone files)
                place_experiment "$temp_file" "$machine" "$timestamp" "" "$path"
                
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
echo "   ./scripts/collect_remote_experiments.sh           # Collect from all machines"
echo "   ./scripts/collect_remote_experiments.sh 1         # Collect from tier 1 (left, mid, right, riight)"
echo "   ./scripts/collect_remote_experiments.sh 2         # Collect from tier 2 (left2, mid2, right2, riight2)"
echo "   ./scripts/collect_remote_experiments.sh 1 left    # Collect only from left machine"
echo "   ./scripts/collect_remote_experiments.sh --adapters 1  # Collect from tier 1 with adapters"
echo "   ./scripts/collect_remote_experiments.sh --adapters    # Collect from all machines with adapters"