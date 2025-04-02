import os
import glob
import json
import re
import argparse
import numpy as np

def parse_median_fitness(fitness_str):
    """
    Given a fitness string like:
      "95% Bootstrap Confidence Interval: (6.2%, 17.2%), Median: 11.7%"
    extract the median as a float (e.g. 11.7). Return np.nan if not found.
    """
    m = re.search(r"Median:\s*([\d\.]+)%", fitness_str)
    if m:
        return float(m.group(1))
    else:
        return np.nan

def load_best_fitness(json_path):
    """
    Load one run's JSON file (a list of up to 30 generation dicts).
    Skip entries where generation == "initial".
    Parse the 'fitness' field for the median, track the maximum across all valid generations.
    Return that maximum. If none found, return np.nan.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    best_fit = np.nan
    for entry in data:
        gen_val = entry.get("generation", None)
        if isinstance(gen_val, str) and gen_val.lower() == "initial":
            continue
        fit_val = parse_median_fitness(entry.get("fitness", ""))
        if np.isnan(best_fit):
            best_fit = fit_val
        else:
            best_fit = max(best_fit, fit_val)
    return best_fit

def process_configuration_folder(folder_path):
    """
    Given a configuration folder that contains exactly three JSON files (one run each),
    load each run and compute the best fitness for that run.
    Return a list of best-fitness values (one per run).
    """
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    run_best = []
    for jf in json_files:
        best_fit = load_best_fitness(jf)
        run_best.append(best_fit)
    return run_best

def parse_folder_name(folder_name):
    """
    We assume the folder name is something like "mgsm_mapadas" or "arc_meta_agent_search".
    The first underscore splits the benchmark from the rest, e.g.:
       "mgsm_mapadas" => (benchmark="mgsm", algorithm="mapadas")
       "arc_meta_agent_search" => (benchmark="arc", algorithm="meta_agent_search")
    If there's no underscore, the entire folder name is the benchmark, and the algorithm is empty.
    """
    parts = folder_name.split('_', 1)
    if len(parts) == 1:
        # No underscore
        return folder_name, ""
    benchmark = parts[0]
    algorithm = parts[1]
    return benchmark, algorithm

def compute_summary_table(data_dir):
    """
    Scan each subfolder in data_dir (each representing a configuration).
    For each folder, we compute the best fitness for each run (3 JSON files).
    Then, group the results by (benchmark, algorithm).
    We'll store a tuple (mean, std) for each (benchmark, algorithm).
    """
    table = {}
    # subfolders in data_dir
    subfolders = [d for d in sorted(os.listdir(data_dir))
                  if os.path.isdir(os.path.join(data_dir, d))]
    for folder in subfolders:
        folder_path = os.path.join(data_dir, folder)
        benchmark, algorithm = parse_folder_name(folder)
        run_best = process_configuration_folder(folder_path)
        arr = np.array(run_best, dtype=float)
        arr = arr[~np.isnan(arr)]  # drop NaNs if any
        if len(arr) == 0:
            mean_val, std_val = np.nan, np.nan
        else:
            mean_val = np.mean(arr)
            std_val = np.std(arr)
        table[(benchmark, algorithm)] = (mean_val, std_val)
    return table

def print_table(table):
    """
    Print an ASCII table with columns = sorted benchmarks, rows = sorted algorithms.
    Each cell is "mean ± std" or "No data".
    """
    # gather all benchmarks, algorithms
    benchmarks = sorted({b for (b, a) in table.keys()})
    algorithms = sorted({a for (b, a) in table.keys()})
    
    col_width = 20
    # header
    header = "Algorithm".ljust(col_width)
    for bmk in benchmarks:
        header += bmk.ljust(col_width)
    print(header)
    print("-" * (col_width * (len(benchmarks) + 1)))

    for alg in algorithms:
        row_str = alg.ljust(col_width)
        for bmk in benchmarks:
            key = (bmk, alg)
            if key in table and not np.isnan(table[key][0]):
                m, s = table[key]
                cell = f"{m:.2f} ± {s:.2f}"
            else:
                cell = "No data"
            row_str += cell.ljust(col_width)
        print(row_str)

def main():
    parser = argparse.ArgumentParser(
        description="Compute average highest fitness and std for each (benchmark, algorithm) from subfolders."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the top-level directory containing config subfolders. Each subfolder has 3 JSON files.")
    args = parser.parse_args()

    table = compute_summary_table(args.data_dir)
    print_table(table)

if __name__ == "__main__":
    main()