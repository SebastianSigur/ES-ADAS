import os
import glob
import json
import argparse
import numpy as np
import csv

def load_run_misproduction_error(json_file):
    """
    Given one run file (a JSON with up to 100 generation entries),
    compute the misproduction error as:
        error = (100 - valid_count) / 100
    where valid_count is how many entries have a "generation" field
    that is not "initial" (case-insensitive). Capped at 100 if it exceeds.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return 1.0  # If we can't read the file, treat it as fully missing (error=1.0)

    if not isinstance(data, list):
        # If it's not a list, we can't parse it as a run of generations
        return 1.0

    valid_count = 0
    for entry in data:
        gen_val = entry.get("generation", None)
        # If 'generation' is missing or is "initial", skip it
        if gen_val is None:
            continue
        # If it's a string and is "initial" (case-insensitive), skip
        if isinstance(gen_val, str) and gen_val.lower() == "initial":
            continue
        # Otherwise, count it as valid
        valid_count += 1

    # Cap at 100
    if valid_count > 100:
        valid_count = 100

    error = (100 - valid_count) / 100.0
    return error

def process_configuration(config_dir):
    """
    For a given configuration folder (which should contain 3 JSON files),
    compute the misproduction error for each run (JSON).
    Returns a list of error values (one per run).
    """
    json_files = sorted(glob.glob(os.path.join(config_dir, "*.json")))
    run_errors = []
    for jf in json_files:
        err = load_run_misproduction_error(jf)
        run_errors.append(err)
    return run_errors

def compute_summary_table(data_dir):
    """
    For each configuration (subdirectory in data_dir), load its 3 JSON files,
    compute the misproduction error for each (one per run), then compute mean and std.
    Returns a list of tuples: (config_name, mean_error, std_error).
    """
    table = []
    # Each config is a subdirectory of data_dir
    config_dirs = [d for d in sorted(os.listdir(data_dir))
                   if os.path.isdir(os.path.join(data_dir, d))]
    for config_name in config_dirs:
        config_path = os.path.join(data_dir, config_name)
        errors = process_configuration(config_path)
        if len(errors) == 0:
            continue
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        table.append((config_name, mean_error, std_error))
    return table

def write_csv(table, output_file):
    """
    Write the summary table to a CSV file with columns:
      Configuration, Mean, StdDev
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Configuration", "Mean", "StdDev"])
        for row in table:
            writer.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}"])

def main():
    parser = argparse.ArgumentParser(
        description="Compute misproduction error for each configuration and export a summary table (CSV)."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing configuration subdirectories.")
    parser.add_argument("--output_file", type=str, default="misproduction_error_table.csv",
                        help="Output CSV filename. If relative, it is saved in data_dir.")
    args = parser.parse_args()

    if not os.path.isabs(args.output_file):
        output_file = os.path.join(args.data_dir, args.output_file)
    else:
        output_file = args.output_file

    table = compute_summary_table(args.data_dir)
    write_csv(table, output_file)
    print(f"Misproduction error table saved to: {output_file}")

if __name__ == "__main__":
    main()