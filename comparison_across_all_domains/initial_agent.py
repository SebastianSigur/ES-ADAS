import os
import json
import re
import numpy as np

def parse_fitness(fitness_str):
    """
    Parses the fitness string to extract the median value.
    Expected format example:
      "95% Bootstrap Confidence Interval: (6.0%, 19.0%), Median: 12.0%"
    Returns:
      The median value as a float.
    """
    # Use regex to extract the number after 'Median:'
    match = re.search(r"Median:\s*([\d\.]+)%", fitness_str)
    if match:
        return float(match.group(1))
    else:
        # If the expected format isn't found, try a simple conversion
        try:
            return float(fitness_str)
        except Exception as e:
            raise ValueError(f"Could not parse fitness string: '{fitness_str}'") from e

def main():
    # List of possible benchmarks. Update if needed.
    benchmarks = ["arc", "drop", "gpqa", "mgsm", "mmlu"]
    
    # Dictionary to accumulate highest initial fitness values for each benchmark
    results = {b: [] for b in benchmarks}
    
    # Loop over all items in the current directory
    for folder in os.listdir("."):
        if not os.path.isdir(folder):
            continue
        
        # We only care about folders ending with "_meta_agent_search"
        if folder.endswith("_meta_agent_search"):
            # Extract the benchmark name by removing "_meta_agent_search"
            bench_name = folder.replace("_meta_agent_search", "")
            
            # If it's one of the known benchmarks, process it
            if bench_name in benchmarks:
                folder_path = os.path.join(".", folder)
                for file_name in os.listdir(folder_path):
                    if file_name.lower().endswith(".json"):
                        file_path = os.path.join(folder_path, file_name)
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
                            continue
                        
                        # Filter for rows corresponding to the "initial" generation.
                        # Adjust this if your JSON structure is different.
                        initial_entries = [
                            entry for entry in data 
                            if entry.get("generation") == "initial"
                        ]
                        
                        if initial_entries:
                            # Parse fitness values using the helper function
                            fitness_vals = [parse_fitness(e["fitness"]) for e in initial_entries]
                            highest_fitness = max(fitness_vals)
                            results[bench_name].append(highest_fitness)
                        else:
                            print(f"No initial generation data in {file_path}")
    
    # Compute mean and standard deviation per benchmark
    for bench in benchmarks:
        fitness_values = results[bench]
        if fitness_values:
            arr = np.array(fitness_values, dtype=float)
            mean_val = arr.mean()
            std_val = arr.std(ddof=1)  # sample standard deviation
            print(f"Benchmark: {bench}")
            print(f"  Mean Highest Initial Generation Fitness: {mean_val:.4f}")
            print(f"  Standard Deviation: {std_val:.4f}\n")
        else:
            print(f"Benchmark: {bench}")
            print("  No data found (or no initial-generation entries).\n")

if __name__ == "__main__":
    main()
