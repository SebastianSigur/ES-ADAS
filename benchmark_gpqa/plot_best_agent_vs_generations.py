import os
import glob
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Helper Functions
# -------------------------
def parse_median_fitness(fitness_str):
    """
    Parse the 'fitness' string and extract the median value as a float.
    Example string:
      "95% Bootstrap Confidence Interval: (5.5%, 16.4%), Median: 10.9%"
    returns: 10.9
    """
    m = re.search(r"Median:\s*([\d\.]+)%", fitness_str)
    if m:
        return float(m.group(1))
    else:
        return np.nan

def load_single_run(filepath, max_generation=100):
    """
    Load one run archive (a JSON file) and compute, for each generation, the cumulative best median fitness.
    For each generation g, it finds the maximum median fitness among all agents at generation g,
    and then takes the maximum up to that point.
    
    Returns a NumPy array of shape (max_generation+1,) containing the cumulative best median fitness per generation.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Dictionary mapping generation -> maximum median fitness found at that generation
    gen_to_fitness = {}
    for entry in data:
        gen_val = entry.get("generation", None)
        if gen_val is None:
            continue
        # Convert "initial" to 0 and other values to int
        if isinstance(gen_val, str):
            if gen_val.lower() == "initial":
                g = 0
            else:
                try:
                    g = int(gen_val)
                except:
                    g = 0
        else:
            g = int(gen_val)
        if g > max_generation:
            continue
        fitness_str = entry.get("fitness", "")
        median_fit = parse_median_fitness(fitness_str)
        # For each generation, store the maximum median fitness observed at that generation.
        if g in gen_to_fitness:
            gen_to_fitness[g] = max(gen_to_fitness[g], median_fit)
        else:
            gen_to_fitness[g] = median_fit

    # Build the cumulative best vector. For each generation, we use the maximum fitness found up to that generation.
    best_values = np.empty(max_generation + 1)
    last_val = np.nan
    for g in range(max_generation + 1):
        if g in gen_to_fitness:
            if np.isnan(last_val):
                last_val = gen_to_fitness[g]
            else:
                last_val = max(last_val, gen_to_fitness[g])
            best_values[g] = last_val
        else:
            best_values[g] = last_val
    return best_values

def load_configuration_runs(config_dir, max_generation=100):
    """
    For a given configuration directory (which should contain 3 JSON run files),
    load each run and return a 2D NumPy array of shape (num_runs, max_generation+1)
    where each row corresponds to the cumulative best fitness values for that run.
    """
    json_files = sorted(glob.glob(os.path.join(config_dir, "*.json")))
    runs = []
    for jf in json_files:
        run_values = load_single_run(jf, max_generation=max_generation)
        runs.append(run_values)
    if not runs:
        return np.array([])
    return np.vstack(runs)

# -------------------------
# Plotting Function
# -------------------------
def plot_configurations(data_dir, output_file, max_generation=100):
    """
    For each configuration (each subdirectory in data_dir), load its runs,
    compute the mean and standard deviation (across runs) of the cumulative best fitness
    per generation, and plot all configurations on one figure.
    """
    # List subdirectories in data_dir
    config_dirs = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    plt.figure(figsize=(10, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for idx, config_path in enumerate(config_dirs):
        config_name = os.path.basename(config_path)
        runs_data = load_configuration_runs(config_path, max_generation=max_generation)
        if runs_data.size == 0:
            print(f"Warning: No run data found for configuration '{config_name}'.")
            continue
        
        # Compute mean and standard deviation across runs (axis=0)
        mean_fit = np.nanmean(runs_data, axis=0)
        std_fit = np.nanstd(runs_data, axis=0)
        generations = np.arange(max_generation + 1)
        
        color = colors[idx % len(colors)]
        labels = {"gpqa_mapadas": "MAPADAS", "gpqa_improved_adas": "Meta Agent Search"}
        plt.plot(generations, mean_fit, label=labels[config_name], color=color, linewidth=2)
        # plt.plot(generations, mean_fit-std_fit,"r--", color=color, linewidth=1,)
        # plt.plot(generations, mean_fit+std_fit,"r--", color=color, linewidth=1,)

        plt.fill_between(generations, mean_fit - std_fit, mean_fit + std_fit,
                         color=color, alpha=0.2)
    
    plt.xlabel("Generations")
    plt.ylabel("Highest Fitness Agent (%)")
    plt.title("MGSM Benchmark")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300)
    plt.close()
    print(f"Plot saved to: {output_file}")

# -------------------------
# Main Function
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot the cumulative highest fitness (best agent) vs. generations for each configuration."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing configuration subdirectories. "
                             "Place this script at the same level as these subdirectories if desired.")
    parser.add_argument("--output_file", type=str, default="best_agents.pdf",
                        help="Filename for the output plot. If relative, it will be saved in the same folder as data_dir.")
    parser.add_argument("--max_generation", type=int, default=100,
                        help="Maximum generation number (default 100).")
    args = parser.parse_args()
    
    # If output_file is a relative path, save it in data_dir
    if not os.path.isabs(args.output_file):
        output_file = os.path.join(args.data_dir, args.output_file)
    else:
        output_file = args.output_file

    plot_configurations(args.data_dir, output_file, max_generation=args.max_generation)

if __name__ == "__main__":
    main()