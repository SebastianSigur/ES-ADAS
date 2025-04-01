import os
import glob
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_median_fitness(fitness_str):
    """
    Extract the median fitness value from a string like:
      "95% Bootstrap Confidence Interval: (13.3%, 27.3%), Median: 20.3%"
    Returns the float value 20.3 or np.nan if not found.
    """
    m = re.search(r"Median:\s*([\d\.]+)%", fitness_str)
    if m:
        return float(m.group(1))
    else:
        return np.nan

def load_avg_fitness(filepath):
    """
    Load one MAP‐Elites JSON file (a dict of up to 12 cells) and compute the average fitness.
    For each cell:
      - If non-null, extract its median fitness from the "fitness" field.
      - If null, assign a value of 0.
    Then, return the average over all 12 cells.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    total_cells = len(data)  # Expected to be 12
    if total_cells == 0:
        return 0.0
    fitness_values = []
    for key, value in data.items():
        if value is not None:
            fit_val = parse_median_fitness(value.get("fitness", ""))
            fitness_values.append(fit_val)
        else:
            fitness_values.append(0.0)
    mean_fit = np.mean(fitness_values)
    return mean_fit

def load_single_run_avg(run_dir, max_generation=100):
    """
    For a given run folder (which contains up to 100 JSON files, one per generation),
    load each JSON file and compute its average fitness (as computed by load_avg_fitness).
    Filenames must include a substring like "_genX" where X is the generation number.
    If a generation is missing, carry forward the last observed value.
    
    Returns a NumPy array of shape (max_generation+1,) with the average fitness per generation.
    """
    json_files = glob.glob(os.path.join(run_dir, "*.json"))
    gen_to_avg = {}
    for filepath in json_files:
        base = os.path.basename(filepath)
        # Extract generation number from filename using pattern "_genX"
        m = re.search(r"_gen(\d+)", base)
        if m:
            gen = int(m.group(1))
            if gen <= max_generation:
                avg_fit = load_avg_fitness(filepath)
                gen_to_avg[gen] = avg_fit
        else:
            continue

    avg_values = np.empty(max_generation + 1)
    last_val = 0.0
    for g in range(max_generation + 1):
        if g in gen_to_avg:
            last_val = gen_to_avg[g]
            avg_values[g] = last_val
        else:
            avg_values[g] = last_val
    return avg_values

def load_configuration_runs_avg(config_dir, max_generation=100):
    """
    For a given configuration directory (which should contain 3 subdirectories, one per run),
    load each run's average fitness vector.
    
    Returns a 2D NumPy array of shape (num_runs, max_generation+1), where each row
    is the average fitness per generation for that run.
    """
    run_dirs = [os.path.join(config_dir, d) for d in sorted(os.listdir(config_dir))
                if os.path.isdir(os.path.join(config_dir, d))]
    runs = []
    for run_dir in run_dirs:
        avg_values = load_single_run_avg(run_dir, max_generation=max_generation)
        runs.append(avg_values)
    if not runs:
        return np.array([])
    return np.vstack(runs)

def plot_configurations_avg(data_dir, output_file, max_generation=100):
    """
    For each configuration (subdirectory in data_dir), load its runs,
    compute the mean and standard deviation (across runs) of the average fitness per generation,
    and plot all configurations on one figure.
    """
    config_dirs = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    plt.figure(figsize=(10, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, config_path in enumerate(config_dirs):
        config_name = os.path.basename(config_path)
        runs_data = load_configuration_runs_avg(config_path, max_generation=max_generation)
        if runs_data.size == 0:
            print(f"Warning: No run data found for configuration '{config_name}'.")
            continue
        
        mean_avg = np.mean(runs_data, axis=0)
        std_avg = np.std(runs_data, axis=0)
        generations = np.arange(max_generation + 1)
        
        color = colors[idx % len(colors)]
        labels = {"config_archive": "includes archive", "config_map": "includes MAP", "config_only_agent": "Base MAP (only selected agent)"}
        plt.plot(generations, mean_avg, label=labels[config_name], color=color, linewidth=2)
        plt.fill_between(generations, mean_avg - std_avg, mean_avg + std_avg,
                         color=color, alpha=0.2)

    plt.xlabel("Generations")
    plt.ylabel("Average Fitness in MAP")
    plt.title("Ablation Past Agents (MGSM Benchmark)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300)
    plt.close()
    print(f"Plot saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot the average fitness vs. generations for each configuration."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing configuration subdirectories. "
                             "Each configuration folder should contain subdirectories (one per run) with up to 100 JSON files each.")
    parser.add_argument("--output_file", type=str, default="avg_fitness.pdf",
                        help="Filename for the output plot. If relative, it will be saved in the same folder as data_dir.")
    parser.add_argument("--max_generation", type=int, default=100,
                        help="Maximum generation number (default 100).")
    args = parser.parse_args()

    if not os.path.isabs(args.output_file):
        output_file = os.path.join(args.data_dir, args.output_file)
    else:
        output_file = args.output_file

    plot_configurations_avg(args.data_dir, output_file, max_generation=args.max_generation)

if __name__ == "__main__":
    main()