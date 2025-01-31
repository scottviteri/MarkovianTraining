import json
import sys
from pathlib import Path

def convert_log_format(input_file):
    # Create output filename by inserting _updated before .log
    input_path = Path(input_file)
    output_file = input_path.parent / f"{input_path.stem}_updated{input_path.suffix}"
    
    new_logs = []
    
    with open(input_file, 'r') as f:
        # First line is config - keep it as is
        config = f.readline()
        new_logs.append(config.strip())
        
        # Process remaining lines
        for line in f:
            if not line.strip():
                continue
                
            entry = json.loads(line)
            
            # Skip if it's a config line
            if "model_type" in entry:
                continue
                
            # Create new format
            new_entry = {
                "Example": {
                    "Question": entry["Prev Observation"].replace("Question: ", ""),
                    "Actor Reasoning": entry["Action"],
                    "Answer": entry["Observation"].replace("Answer: ", ""),
                },
                "Training Metrics": {
                    "Normalized Reward": entry["Normalized Reward"],
                    "Avg Log Prob": entry["Avg Log Prob"],
                    "Baseline Avg Log Prob": entry.get("Baseline Avg Log Prob", 0),
                },
                "Batch Index": entry["Batch Index"]
            }
            
            new_logs.append(json.dumps(new_entry))

    # Write to new file
    with open(output_file, 'w') as f:
        for line in new_logs:
            f.write(line + '\n')
            
    print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_log_format.py <input_log_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    convert_log_format(input_file)