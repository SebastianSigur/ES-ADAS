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
    Returns float(20.3) or np.nan if not found.
    """
    m = re.search(r"Median:\s*([\d\.]+)%", fitness_str)
    if m:
        return float(m.group(1))
    else:
        return np.nan

def load_qd_score(filepath):
    """
    Loads one MAP‐Elites JSON file (dict of up to 12 cells).
    For each of the 12 cells:
      - If cell is non-null, parse the median fitness from the "fitness" field.
      - If cell is null, assign 0.0.
    coverage = (# non-null cells)/12
    mean_fitness = average of all 12 values (non-null => median, null => 0)
    QD = coverage * mean_fitness
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    total_cells = len(data)  # Expect 12
    if total_cells == 0:
        return 0.0
    filled_count = 0
    fitness_values = []
    for key, value in data.items():
        if value is not None:
            filled_count += 1
            fit_val = parse_median_fitness(value.get("fitness", ""))
            fitness_values.append(fit_val)
        else:
            # Null cell => treat as 0
            fitness_values.append(0.0)
    coverage = filled_count / total_cells
    mean_fit = np.mean(fitness_values)
    return coverage * mean_fit

def load_single_run_qd(run_dir, max_generation=100):
    """
    For a given run folder, find up to 100 JSON files named like "..._genX.json"
    to identify generation X. For each generation, compute the QD score
    and store it in a vector of length (max_generation+1). Missing generations
    carry forward the last value.
    """
    json_files = glob.glob(os.path.join(run_dir, "*.json"))
    gen_to_qd = {}
    for filepath in json_files:
        base = os.path.basename(filepath)
        # Look for something like "_gen4"
        m = re.search(r"_gen(\d+)", base)
        if m:
            gen = int(m.group(1))
            if gen <= max_generation:
                qd = load_qd_score(filepath)
                gen_to_qd[gen] = qd
        else:
            # If the file doesn't have "_gen(\d+)", skip it
            continue

    qd_values = np.empty(max_generation + 1)
    last_val = 0.0
    for g in range(max_generation + 1):
        if g in gen_to_qd:
            last_val = gen_to_qd[g]
            qd_values[g] = last_val
        else:
            qd_values[g] = last_val
    return qd_values

def load_configuration_runs_qd(config_dir, max_generation=100):
    """
    For each configuration, we expect multiple run subdirectories. 
    Each run subdirectory has up to 100 JSON files with "_genX" in the name.
    Return a 2D array shape (num_runs, max_generation+1).
    """
    run_dirs = [os.path.join(config_dir, d) for d in sorted(os.listdir(config_dir))
                if os.path.isdir(os.path.join(config_dir, d))]
    runs = []
    for run_dir in run_dirs:
        qd_values = load_single_run_qd(run_dir, max_generation=max_generation)
        runs.append(qd_values)
    if not runs:
        return np.array([])
    return np.vstack(runs)

def plot_configurations_qd(data_dir, output_file, max_generation=100):
    """
    For each configuration (subdirectory in data_dir), load runs,
    compute mean & std of QD over generations, and plot them.
    """
    config_dirs = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    plt.figure(figsize=(10, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, config_path in enumerate(config_dirs):
        config_name = os.path.basename(config_path)
        runs_data = load_configuration_runs_qd(config_path, max_generation=max_generation)
        if runs_data.size == 0:
            print(f"Warning: No run data found for configuration '{config_name}'.")
            continue

        mean_qd = np.mean(runs_data, axis=0)
        std_qd = np.std(runs_data, axis=0)
        generations = np.arange(max_generation + 1)

        color = colors[idx % len(colors)]
        labels = {"config_archive": "includes archive", "config_map": "includes MAP", "config_only_agent": "Base MAP (only selected agent)"}
        plt.plot(generations, mean_qd, label=labels[config_name], color=color, linewidth=2)
        plt.fill_between(generations, mean_qd - std_qd, mean_qd + std_qd,
                         color=color, alpha=0.2)

    plt.xlabel("Generations")
    plt.ylabel("QD Score (Coverage × Mean Fitness)")
    plt.title("Ablation Past Agents (MGSM Benchmark)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300)
    plt.close()
    print(f"Plot saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot QD score across generations for each configuration. "
                    "Filenames must include '_genX' to parse generation."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to the directory containing config subfolders. "
                             "Each config subfolder has run subfolders, each run has JSONs named like '*_gen4.json'.")
    parser.add_argument("--output_file", default="qd_score.pdf",
                        help="Output filename (PDF recommended). If relative, it's placed in data_dir.")
    parser.add_argument("--max_generation", type=int, default=100,
                        help="Maximum generation to plot (default=100).")
    args = parser.parse_args()

    if not os.path.isabs(args.output_file):
        output_file = os.path.join(args.data_dir, args.output_file)
    else:
        output_file = args.output_file

    plot_configurations_qd(args.data_dir, output_file, max_generation=args.max_generation)

if __name__ == "__main__":
    main()