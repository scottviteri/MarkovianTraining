#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def moving_average(data, window_size):
    """Calculate moving average, properly handling NaN values"""
    if len(data) < window_size:
        return data
        
    # Convert the data to a numpy array to ensure correct handling of NaN values
    data_array = np.array(data, dtype=float)
    
    # Use a technique that doesn't count NaN values in the average
    result = np.zeros(len(data_array) - window_size + 1)
    
    for i in range(len(result)):
        window = data_array[i:i+window_size]
        # Count only non-NaN values
        valid_values = window[~np.isnan(window)]
        if len(valid_values) > 0:
            result[i] = np.mean(valid_values)
        else:
            result[i] = np.nan
    
    return result

def get_nested_value(entry, path):
    """Helper function to get nested dictionary values using dot notation"""
    value = entry
    for key in path.split("."):
        if value is None or key not in value:
            return np.nan
        value = value[key]
    
    # Handle special string indicators like "NaN (no active examples)"
    if isinstance(value, str) and "NaN" in value:
        return np.nan
        
    return value

def gp_smooth_with_confidence(x, y, sigma=30.0, confidence_scale=0.05):
    """Apply Gaussian process-like smoothing with confidence bands"""
    # Apply Gaussian smoothing to the main line
    smoothed_y = gaussian_filter1d(y, sigma=sigma)
    
    # Calculate local variance for confidence bands
    # Use a sliding window to estimate local noise
    window_size = min(50, len(y) // 10)
    local_std = np.zeros_like(y)
    
    for i in range(len(y)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(y), i + window_size // 2)
        window_data = y[start_idx:end_idx]
        window_smooth = smoothed_y[start_idx:end_idx]
        
        # Calculate residuals in the window
        residuals = window_data - window_smooth
        local_std[i] = np.std(residuals) if len(residuals) > 1 else confidence_scale
    
    # Smooth the standard deviation as well
    smoothed_std = gaussian_filter1d(local_std, sigma=sigma//2)
    
    # Scale the confidence bands
    confidence_bands = smoothed_std * confidence_scale * 50  # Adjust multiplier for visibility
    
    return smoothed_y, confidence_bands

def load_and_plot_experiment(file_path, label, color, window_size=30):
    """Load experiment data and return x, y data for plotting"""
    print(f"Loading {file_path}...")
    
    with open(file_path, "r") as f:
        file_contents = f.readlines()
        hyperparameters = json.loads(file_contents[0].strip())
        entries = [json.loads(line) for line in file_contents[1:]]
    
    # Extract normalized reward data
    raw_data = [
        get_nested_value(entry, "Training Metrics.Normalized Reward")
        for entry in entries
    ]
    
    # Filter out None values and convert to float array
    valid_data = []
    for d in raw_data:
        if d is None:
            valid_data.append(np.nan)
        else:
            try:
                valid_data.append(float(d))
            except (ValueError, TypeError):
                valid_data.append(np.nan)
    
    # Only proceed if we have valid data
    if valid_data and not all(np.isnan(d) for d in valid_data):
        # Convert to numpy array for handling NaN values
        data_array = np.array(valid_data, dtype=float)
        
        # First apply moving average for initial smoothing
        initially_smoothed = moving_average(data_array, window_size)
        
        # Create x-coordinates for smoothed data
        offset = (window_size - 1) // 2 if window_size > 1 else 0
        x_coords = np.arange(offset, offset + len(initially_smoothed))
        
        # Filter out NaN values
        mask = ~np.isnan(initially_smoothed)
        if np.any(mask):
            clean_x = x_coords[mask]
            clean_y = initially_smoothed[mask]
            
            # Apply Gaussian process-like smoothing with confidence bands
            if len(clean_y) > 50:  # Only apply GP smoothing if we have enough points
                gp_smooth, confidence = gp_smooth_with_confidence(clean_x, clean_y)
                return clean_x, gp_smooth, confidence, hyperparameters
            else:
                # Fallback for short series
                return clean_x, clean_y, np.ones_like(clean_y) * 0.01, hyperparameters
    
    return None, None, None, None

def main():
    # Define the log files and their labels
    experiments = [
        ("results/wiki_continuation/20250801_194419_left/log.jsonl", "left1", "#1f77b4"),
        ("results/wiki_continuation/20250801_194437_mid2/log.jsonl", "mid2", "#ff7f0e"), 
        ("results/wiki_continuation/20250801_194450_right2/log.jsonl", "right2", "#2ca02c"),
        ("results/wiki_continuation/20250801_194514_riight2/log.jsonl", "riight2", "#d62728")
    ]
    
    window_size = 30  # Increased for more smoothing
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    max_x = 0
    valid_experiments = []
    
    # Process each experiment
    for file_path, label, color in experiments:
        if os.path.exists(file_path):
            x_coords, y_smooth, confidence, hyperparams = load_and_plot_experiment(file_path, label, color, window_size)
            if x_coords is not None and y_smooth is not None:
                # Plot the main smoothed line
                plt.plot(x_coords, y_smooth, label=label, color=color, linewidth=3, alpha=0.9)
                
                # Plot confidence bands (Gaussian process-like blur)
                plt.fill_between(x_coords, 
                               y_smooth - confidence, 
                               y_smooth + confidence,
                               color=color, alpha=0.15, linewidth=0)
                
                # Add a slightly thicker semi-transparent line for more blur effect
                plt.plot(x_coords, y_smooth, color=color, linewidth=6, alpha=0.2)
                
                max_x = max(max_x, np.max(x_coords))
                valid_experiments.append((label, hyperparams))
                print(f"Successfully plotted {label} with {len(x_coords)} points")
            else:
                print(f"Warning: No valid data found for {label}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Set up the plot formatting similar to plot_training_metrics.py
    plt.xlabel("Training Batch No. []", fontsize=14)
    plt.ylabel("ln π(ans|cot) - ln π(ans|cot') []", fontsize=14)
    plt.title("Normalized Reward Comparison (GP-Style Smoothing)", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.legend(fontsize=12, framealpha=0.9)
    
    # Set x-axis limit to the maximum x value
    if max_x > 0:
        plt.xlim(0, max_x)
    
    # Add smoothing info
    plt.text(
        0.95, 0.05,
        f"Moving avg window = {window_size}\nGaussian process-style smoothing",
        transform=plt.gca().transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=10,
        bbox=dict(
            facecolor='white',
            alpha=0.9,
            edgecolor='black',
            pad=5,
            boxstyle='round,pad=0.5'
        )
    )
    
    # Improve overall aesthetics
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Tight layout and save
    plt.tight_layout()
    output_file = "combined_normalized_reward_gp_smoothed.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    # Print experiment info
    print(f"\nSuccessfully plotted {len(valid_experiments)} experiments:")
    for label, hyperparams in valid_experiments:
        cot_length = hyperparams.get('cot_length', 'N/A')
        temperature = hyperparams.get('temperature', 'N/A')
        model_type = hyperparams.get('model_type', 'N/A')
        print(f"  {label}: model={model_type}, cot_length={cot_length}, temp={temperature}")

if __name__ == "__main__":
    main()