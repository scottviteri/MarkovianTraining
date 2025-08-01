import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import argparse
import math
from constants import EI_SKIP_INITIAL
import sys
import glob


def get_latest_log_file():
    """
    Find the most recent log.jsonl file in any results subdirectory.
    
    Returns:
        str: Path to the most recent log file, or None if no log files found
    """
    # Look for any log.jsonl file in subdirectories of results/
    log_files = glob.glob("results/**/log.jsonl", recursive=True)
    
    if not log_files:
        return None
    
    # Sort by modification time, most recent first
    return sorted(log_files, key=os.path.getmtime, reverse=True)[0]


def get_nested_value(entry, path, metrics_dict=None):
    """Helper function to get nested dictionary values using dot notation, with optional derived metrics"""
    # Handle special cases for missing metrics
    if path == "Training Metrics.Active Only Loss" or path == "Training Metrics.Active Only PG Loss":
        # Check if EI is enabled in this log entry
        if "EI Metrics" not in entry:
            return np.nan
    
    if path == "Training Metrics.Active Samples.Fraction":
        # Check if Active Samples metrics exist (EI enabled)
        if "Training Metrics" not in entry or "Active Samples" not in entry.get("Training Metrics", {}):
            return np.nan
    
    if path == "Training Metrics.Critic Answer Log Probs" and metrics_dict is not None:
        # Calculate Critic Answer Log Probs if not directly available
        actor_probs = get_nested_value(entry, "Training Metrics.Actor Answer Log Probs")
        norm_reward = get_nested_value(entry, "Training Metrics.Normalized Reward")
        if actor_probs is not None and norm_reward is not None:
            return actor_probs - norm_reward
        return np.nan
    
    # Handle computed actor reward metrics
    if path == "computed_loss_balance":
        pg_loss = get_nested_value(entry, "Training Metrics.Policy Gradient Loss")
        reward_loss = get_nested_value(entry, "Actor Reward Metrics.Reward Gradient Loss")
        if pg_loss is not None and reward_loss is not None:
            return compute_loss_balance_score(pg_loss, reward_loss)
        return np.nan
    
    if path == "computed_sign_agreement":
        pg_loss = get_nested_value(entry, "Training Metrics.Policy Gradient Loss")
        reward_loss = get_nested_value(entry, "Actor Reward Metrics.Reward Gradient Loss")
        if pg_loss is not None and reward_loss is not None:
            return 1.0 if np.sign(pg_loss) == np.sign(reward_loss) else 0.0
        return np.nan
    
    # Handle actor reward metrics that might not exist in older logs
    if path.startswith("Actor Reward Metrics.") and "Actor Reward Metrics" not in entry:
        return np.nan

    # Standard nested dictionary access
    value = entry
    for key in path.split("."):
        if value is None or key not in value:
            return np.nan
        value = value[key]
    
    # Handle special string indicators like "NaN (no active examples)"
    if isinstance(value, str) and "NaN" in value:
        return np.nan
    
    # Handle infinite values in PG vs Reward Ratio - convert to NaN for better plotting
    if path == "Actor Reward Metrics.PG vs Reward Ratio" and (value == "inf" or value == float('inf') or value == float('-inf')):
        return np.nan
        
    return value


def compute_loss_balance_score(pg_loss, reward_loss):
    """Compute a normalized loss balance score that avoids division by zero."""
    return (pg_loss - reward_loss) / np.maximum(np.abs(pg_loss) + np.abs(reward_loss), 1e-10)


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


def add_hyperparameters_display(fig, hyperparameters):
    """Add a text box showing all hyperparameters at the bottom of the figure like a caption."""
    # Format hyperparameters for display
    param_text = format_hyperparameters_text(hyperparameters)
    
    # Calculate appropriate spacing based on number of parameter lines
    param_lines = param_text.count('\n') + 1
    # Reserve generous space at the bottom for hyperparameters - scale with number of lines
    # Increased margins to prevent overlap with bottom plots
    bottom_margin = max(0.25, 0.15 + param_lines * 0.025)
    fig.subplots_adjust(bottom=bottom_margin)
    
    # Add text box at the bottom center of the figure like a caption
    fig.text(0.5, 0.02, param_text, fontsize=7, 
             verticalalignment='bottom', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9),
             family='monospace')


def format_hyperparameters_text(hyperparameters):
    """Format hyperparameters dictionary into a readable multi-line string."""
    # Group related parameters for better organization
    training_params = {}
    model_params = {}
    task_params = {}
    other_params = {}
    
    # Categorize parameters
    for key, value in hyperparameters.items():
        if key in ['lr', 'batch_size', 'num_batches', 'use_ppo', 'use_ei', 'normalize_loss', 
                   'parallel', 'kl_penalty', 'temperature', 'entropy_bonus']:
            training_params[key] = value
        elif key in ['model_type', 'model_name', 'cot_length', 'lora_rank', 'lora_alpha']:
            model_params[key] = value
        elif key in ['task_type', 'num_examples_per_task', 'r']:
            task_params[key] = value
        else:
            other_params[key] = value
    
    # Build formatted text with shorter lines to prevent cutoff
    lines = ["Hyperparameters:"]
    
    def format_line(params, max_chars=80):
        """Format parameters into lines that don't exceed max_chars"""
        if not params:
            return []
        
        param_strs = [f"{k}={v}" for k, v in params.items()]
        lines = []
        current_line = ""
        
        for param_str in param_strs:
            test_line = current_line + (" | " if current_line else "") + param_str
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = param_str
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    if training_params:
        train_lines = format_line(training_params)
        for i, line in enumerate(train_lines):
            prefix = "Training: " if i == 0 else "         "
            lines.append(prefix + line)
    
    if model_params:
        model_lines = format_line(model_params)
        for i, line in enumerate(model_lines):
            prefix = "Model: " if i == 0 else "       "
            lines.append(prefix + line)
    
    if task_params:
        task_lines = format_line(task_params)
        for i, line in enumerate(task_lines):
            prefix = "Task: " if i == 0 else "      "
            lines.append(prefix + line)
    
    if other_params:
        other_lines = format_line(other_params)
        for i, line in enumerate(other_lines):
            prefix = "Other: " if i == 0 else "       "
            lines.append(prefix + line)
    
    return "\n".join(lines)


def plot_combined_metrics(file_paths, host_names, window_size=10, output_file=None, plot_summary=False, max_index=None, average=False, show_std=False, show_legend=True, label_size=10, show_title=True, show_raw=True):
    """Plot metrics from multiple files on the same plot.
    
    Args:
        file_paths: List of paths to log files
        host_names: List of host names for legend labels
        window_size: Size of moving average window for smoothing (default: 10)
        output_file: Path for output plot file
        plot_summary: Whether to plot summary metrics only (varies by task)
        max_index: Maximum batch index to plot (truncates data)
        average: Whether to average values across input files
        show_std: Whether to show standard deviation bands when averaging
        show_legend: Whether to show legend in plots
        label_size: Font size for labels
        show_title: Whether to show titles on plots
        show_raw: Whether to show both raw and smoothed versions (default: True)
    """
    # Read first file to get task type and check available metrics
    with open(file_paths[0], "r") as f:
        file_contents = f.readlines()
        hyperparameters = json.loads(file_contents[0].strip())
        first_entry = json.loads(file_contents[1].strip())  # Read first log entry
    
    task_type = hyperparameters.get('task_type', 'unknown')
    has_answer_logprobs = "Actor Answer Log Probs" in first_entry.get("Training Metrics", {})
    has_critic_probs = "Critic Answer Log Probs" in first_entry.get("Training Metrics", {})
    
    # Check if important features are enabled
    normalize_loss = hyperparameters.get('normalize_loss', True)
    ei_enabled = hyperparameters.get('use_ei') is not None
    actor_reward_weight = hyperparameters.get('actor_reward_weight', 0.0)
    actor_rewards_enabled = actor_reward_weight > 0.0
    has_actor_reward_metrics = "Actor Reward Metrics" in first_entry
    
    # Initialize metrics_dict for deriving critic probs if needed
    metrics_dict = None if has_critic_probs else {"derive_critic": True}
    
    if output_file is None:
        output_file = f"combined_metrics_{task_type}.png"
    
    if plot_summary:
        # For arithmetic tasks or gsm8k
        if task_type in ['arithmetic', 'gsm8k']:
            metrics_to_plot = [
                ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Training Batch No. []", "ln π(ans|cot)"),
                ("Example.Contains Answer", "Contains Answer", "Training Batch No. []", "Fraction")
            ]
            
            # Add critic metrics only if normalization is enabled
            if normalize_loss and has_critic_probs:
                metrics_to_plot.insert(1, ("Training Metrics.Critic Answer Log Probs", "Critic Answer Log Probs", "Training Batch No. []", "ln π'(ans|cot)"))
                
        # For wiki tasks
        elif task_type.startswith('wiki_'):
            if has_answer_logprobs:
                metrics_to_plot = [
                    ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')"),
                    ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Training Batch No. []", "ln π(ans|cot)")
                ]
                
                # Add critic metrics only if normalization is enabled
                if normalize_loss and has_critic_probs:
                    metrics_to_plot.append(("Training Metrics.Critic Answer Log Probs", "Critic Answer Log Probs", "Training Batch No. []", "ln π'(ans|cot)"))
            else:
                metrics_to_plot = [
                    ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')")
                ]
        else:
            metrics_to_plot = [
                ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')")
            ]
    else:
        # Start with basic metrics that are always present
        base_metrics = [
            ("Training Metrics.Loss", "Total Loss (All Examples)", "Training Batch No. []", "Loss"),
            ("Training Metrics.Policy Gradient Loss", "PG Loss (All Examples)", "Training Batch No. []", "Loss"),
            ("Training Metrics.KL", "KL Divergence", "Training Batch No. []", "KL"),
            ("Training Metrics.Gradient Norm", "Gradient Norm", "Training Batch No. []", "Norm"),
            ("Training Metrics.Advantage", "Advantage", "Training Batch No. []", "Value"),
            ("Training Metrics.Normalized Reward", "Normalized Reward", "Training Batch No. []", "ln π(ans|cot) - ln π(ans|cot')")
        ]
        
        # Add EI-specific metrics only if EI is enabled
        if ei_enabled:
            base_metrics.extend([
                ("Training Metrics.Active Only Loss", "Total Loss (Active Examples)", "Training Batch No. []", "Loss"),
                ("Training Metrics.Active Only PG Loss", "PG Loss (Active Examples)", "Training Batch No. []", "Loss"),
                ("Training Metrics.Active Samples.Fraction", "Fraction of Active Samples", "Training Batch No. []", "Fraction")
            ])
        
        # Add actor log probs if available
        if has_answer_logprobs:
            base_metrics.append(
                ("Training Metrics.Actor Answer Log Probs", "Actor Answer Log Probs", "Training Batch No. []", "ln π(ans|cot)")
            )
            
            # Add critic log probs only if normalization is enabled and they're available
            if normalize_loss and has_critic_probs:
                base_metrics.append(
                ("Training Metrics.Critic Answer Log Probs", "Critic Answer Log Probs", "Training Batch No. []", "ln π'(ans|cot)")
                )
        
        # Add Contains Answer metric for GSM8K
        if task_type == "gsm8k":
            base_metrics.append(
                ("Example.Contains Answer", "Contains Answer", "Training Batch No. []", "Fraction")
            )
        
        # Add actor reward specific metrics if enabled
        if actor_rewards_enabled and has_actor_reward_metrics:
            base_metrics.extend([
                ("Training Metrics.Policy Gradient Loss", "Policy Gradient Loss", "Training Batch No. []", "PG Loss"),
                ("Actor Reward Metrics.Reward Gradient Loss", "Reward Gradient Loss", "Training Batch No. []", "∇_θ R_θ(τ)"),
                ("computed_loss_balance", "Loss Balance Score", "Training Batch No. []", "Normalized Difference"),
                ("computed_sign_agreement", "Loss Sign Agreement", "Training Batch No. []", "Same Direction (0/1)")
            ])
            
        metrics_to_plot = base_metrics

    # Calculate required number of rows and columns for subplots
    if len(metrics_to_plot) > 1:
        num_cols = 2
        num_rows = math.ceil(len(metrics_to_plot) / 2)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
        axs = np.array(axs).reshape(-1)
    else:
        num_rows, num_cols = 1, 1
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))
        axs = np.array([axs])

    axs = axs.flatten()

    for metric_idx, (metric_path, title, xlabel, ylabel, *extra) in enumerate(
        metrics_to_plot
    ):
        all_data = []  # Store data from all files for averaging
        
        for file_path, host_name in zip(file_paths, host_names):
            with open(file_path, "r") as f:
                file_contents = f.readlines()
            
            hyperparameters = json.loads(file_contents[0].strip())
            label = f"c{hyperparameters.get('cot_length', 400)}t{hyperparameters.get('temperature',1.0)}ei{hyperparameters['use_ei']}kl{hyperparameters.get('kl_penalty', 'NA')}"
            
            entries = [json.loads(line) for line in file_contents[1:]]
            if max_index is not None:
                entries = entries[:max_index]
            
            # Determine skip amount based on whether EI is enabled
            skip_initial = EI_SKIP_INITIAL if hyperparameters.get('use_ei') is not None else 0
            
            # Extract data points, handling None values, NaN, and strings
            raw_data = [
                get_nested_value(entry, metric_path, metrics_dict)
                for entry in entries[skip_initial:]
            ]
            
            # Filter out None values and convert to float array
            valid_data = []
            for d in raw_data:
                if d is None:
                    valid_data.append(np.nan)  # Convert None to NaN for proper handling
                elif isinstance(d, str) and "NaN" in d:
                    valid_data.append(np.nan)  # Handle "NaN" strings
                else:
                    # Handle special cases for certain metrics
                    if metric_path == "Example.Contains Answer":
                        valid_data.append(1 if d else 0)
                    else:
                        try:
                            valid_data.append(float(d))
                        except (ValueError, TypeError):
                            valid_data.append(np.nan)
            
            # Only proceed if we have valid data
            if valid_data and not all(np.isnan(d) for d in valid_data):
                if average:
                    all_data.append(valid_data)
                else:
                    # Convert to numpy array for handling NaN values
                    data_array = np.array(valid_data, dtype=float)
                    
                    # Create x-coordinates for raw data
                    x_coords_raw = np.arange(skip_initial, skip_initial + len(data_array))
                    
                    # Plot raw data with transparency (only if show_raw is enabled)
                    if show_raw:
                        mask_raw = ~np.isnan(data_array)
                        if np.any(mask_raw):  # Only plot if we have any valid points
                            axs[metric_idx].plot(
                                x_coords_raw[mask_raw],
                                data_array[mask_raw],
                                alpha=0.3,
                                linewidth=1,
                                label=f"{label} (Raw)",
                                linestyle='-'
                            )
                    
                    # Smooth the data, properly handling NaN values
                    smoothed_data = moving_average(data_array, window_size)
                    
                    # Create x-coordinates for smoothed data, accounting for the window size
                    offset = (window_size - 1) // 2 if window_size > 1 else 0
                    x_coords = np.arange(skip_initial + offset, skip_initial + offset + len(smoothed_data))
                    
                    # Filter out NaN values before plotting smoothed data
                    mask = ~np.isnan(smoothed_data)
                    if np.any(mask):  # Only plot if we have any valid points
                        # Adjust label based on whether raw data is shown
                        smooth_label = f"{label} (Smoothed)" if show_raw else label
                        axs[metric_idx].plot(
                            x_coords[mask],
                            smoothed_data[mask],
                            linewidth=2,
                            label=smooth_label,
                            linestyle='-'
                        )
        
        if average and all_data and any(len(d) > 0 for d in all_data):
            # Convert all data to float arrays
            all_data = [np.array(d, dtype=float) for d in all_data]
            
            # Pad shorter sequences with NaN
            max_len = max(len(d) for d in all_data)
            padded_data = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in all_data]
            padded_array = np.array(padded_data)
            
            # Calculate mean and std, ignoring NaN values
            mean_data = np.nanmean(padded_array, axis=0)
            std_data = np.nanstd(padded_array, axis=0) if show_std else None
            
            # Remove trailing NaN values
            valid_mask = ~np.isnan(mean_data)
            if np.any(valid_mask):  # Only continue if we have valid data
                mean_data = mean_data[valid_mask]
                if show_std:
                    std_data = std_data[valid_mask]
                
                # Create x-coordinates for raw averaged data
                x_range_raw = np.arange(skip_initial, skip_initial + len(mean_data))
                
                # Plot raw averaged data with transparency (only if show_raw is enabled)
                if show_raw:
                    mask_raw = ~np.isnan(mean_data)
                    if np.any(mask_raw):  # Only plot if we have any valid points
                        axs[metric_idx].plot(
                            x_range_raw[mask_raw],
                            mean_data[mask_raw],
                            alpha=0.3,
                            linewidth=1,
                            label="Average (Raw)",
                            color='blue',
                            linestyle='-'
                        )
                        
                        # Add raw std bands if requested
                        if show_std:
                            std_data_filtered = std_data[mask_raw]
                            axs[metric_idx].fill_between(
                                x_range_raw[mask_raw],
                                mean_data[mask_raw] - std_data_filtered,
                                mean_data[mask_raw] + std_data_filtered,
                                alpha=0.1,
                                color='blue',
                                label='±1 SD (Raw)'
                            )
                
                # Smooth mean and std data
                smoothed_mean = moving_average(mean_data, window_size)
                if show_std:
                    smoothed_std = moving_average(std_data, window_size)
                
                # Create x-coordinates for smoothed data
                offset = (window_size - 1) // 2 if window_size > 1 else 0
                x_range = np.arange(skip_initial + offset, skip_initial + offset + len(smoothed_mean))
                
                # Filter out NaN values before plotting smoothed data
                mask = ~np.isnan(smoothed_mean)
                if np.any(mask):  # Only plot if we have any valid points
                    # Adjust label based on whether raw data is shown
                    smooth_label = "Average (Smoothed)" if show_raw else "Average"
                    # Plot smoothed mean line
                    line = axs[metric_idx].plot(
                        x_range[mask],
                        smoothed_mean[mask],
                        linewidth=2,
                        label=smooth_label,
                        color='blue',
                        linestyle='-'
                    )[0]
                    
                    # Add smoothed std bands if requested
                    if show_std:
                        # Ensure std data is aligned with mean data
                        smoothed_std_filtered = smoothed_std[mask]
                        std_label = '±1 SD (Smoothed)' if show_raw else '±1 SD'
                        axs[metric_idx].fill_between(
                            x_range[mask],
                            smoothed_mean[mask] - smoothed_std_filtered,
                            smoothed_mean[mask] + smoothed_std_filtered,
                            alpha=0.2,
                            color=line.get_color(),
                            label=std_label
                        )

        # Set y-limits for Contains Answer plot
        if title == "Contains Answer":
            axs[metric_idx].set_ylim(-0.05, 1.05)

        # Only add legend for specific conditions
        if len(axs[metric_idx].get_lines()) > 0 and show_legend:
            # For GSM8K summary plots, only show legend on the rightmost plot
            if not (plot_summary and task_type == "gsm8k") or metric_idx == len(metrics_to_plot) - 1:
                axs[metric_idx].legend(fontsize=label_size)
            
        # Set labels and their sizes
        axs[metric_idx].set_xlabel(xlabel, fontsize=label_size)
        axs[metric_idx].set_ylabel(ylabel, fontsize=label_size)
        # Set tick label sizes
        axs[metric_idx].tick_params(axis='both', which='major', labelsize=label_size)
        # Add grid with light gray lines
        axs[metric_idx].grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Add smoothing window info in bottom right corner
        axs[metric_idx].text(
            0.95, 0.05,  # Changed from 0.98, 0.02
            f"Smoothing window = {window_size}",
            transform=axs[metric_idx].transAxes,
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=label_size * 0.8,
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='black',
                pad=3,
                boxstyle='round,pad=0.5'
            )
        )
        
        if extra:
            for key, value in extra[0].items():
                getattr(axs[metric_idx], f"set_{key}")(value)

        # Set title if enabled
        if show_title:
            axs[metric_idx].set_title(title, fontsize=label_size)

    # Remove any unused subplots
    for i in range(len(metrics_to_plot), len(axs)):
        fig.delaxes(axs[i])

    # Add hyperparameters display at the top
    add_hyperparameters_display(fig, hyperparameters)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Combined plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Window size for moving average (default: 10)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output plot file path (optional)",
    )
    parser.add_argument(
        "--plot_summary",
        action="store_true",
        help="Plot summary metrics (varies by task type)",
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        type=str,
        help="Direct paths to log files to plot",
    )
    parser.add_argument(
        "indices",
        nargs="*",
        type=int,
        help="Indices of machines to include (1-based indexing)",
    )
    parser.add_argument(
        "--max_index",
        type=int,
        help="Maximum index to plot (truncates data after this point)",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help="Average the values across all input files",
    )
    parser.add_argument(
        "--show_std",
        action="store_true",
        help="Show standard deviation bands when averaging",
    )
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Don't show the legend in the plots",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=10,
        help="Font size for all labels (axis labels, tick labels, and legend)",
    )
    parser.add_argument(
        "--no_title",
        action="store_true",
        help="Don't show titles on the plots",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Show only smoothed versions (default: show both raw and smoothed)",
    )
    args = parser.parse_args()

    if args.files:
        # Use directly specified files
        files_to_plot = []
        host_names_to_plot = []
        valid_files = False

        for file_path in args.files:
            if os.path.exists(file_path):
                files_to_plot.append(file_path)
                # Use filename as host name for the legend
                host_names_to_plot.append(os.path.basename(os.path.dirname(file_path)))
                valid_files = True
            else:
                print(f"Warning: Could not find log file: {file_path}")
    elif len(args.indices) > 0:
        # Define the list of hosts (keep in sync with download.sh)
        hosts = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
        ]
        # Collect files and corresponding host names to plot
        files_to_plot = []
        host_names_to_plot = []
        valid_files = False

        for i in args.indices:
            if i < 1 or i > len(hosts):
                print(f"Invalid index: {i} (must be between 1 and {len(hosts)})")
                continue
            hostname = hosts[i - 1].split(":")[0]  # Remove port if present
            log_path = f"./results_{hostname}/log.jsonl"
            if os.path.exists(log_path):
                files_to_plot.append(log_path)
                host_names_to_plot.append(hosts[i - 1])
                valid_files = True
            else:
                print(f"Warning: Could not find log file for index {i} ({log_path})")
    else:
        # Single file plotting (use latest log file)
        log_file = get_latest_log_file()
        if log_file is None:
            print("Error: No log files found in results directory")
            sys.exit(1)
        files_to_plot = [log_file]
        host_names_to_plot = ["local"]
        valid_files = True

    if valid_files:
        plot_combined_metrics(
            files_to_plot,
            host_names_to_plot,
            window_size=args.window_size,
            output_file=args.output_file,
            plot_summary=args.plot_summary,
            max_index=args.max_index,
            average=args.average,
            show_std=args.show_std,
            show_legend=not args.no_legend,
            label_size=args.label_size,
            show_title=not args.no_title,
            show_raw=not args.no_raw,
        )
    else:
        print("No valid files found to plot")
        sys.exit(1)
