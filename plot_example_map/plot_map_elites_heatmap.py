import os
import json
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_median_fitness(fitness_str):
    """
    Given a fitness string like:
      "95% Bootstrap Confidence Interval: (6.2%, 17.2%), Median: 11.7%"
    extract and return the median as a float (e.g. 11.7).
    If no median is found, return np.nan.
    """
    match = re.search(r"Median:\s*([\d\.]+)%", fitness_str)
    if match:
        return float(match.group(1))
    else:
        return np.nan

def load_json_file_from_folder(data_dir):
    """
    Looks for any JSON file in the specified folder and returns its full path.
    If more than one is found, returns the first one.
    """
    json_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No JSON file found in {data_dir}")
    return os.path.join(data_dir, json_files[0])

def build_heatmap_matrix(json_file):
    """
    Loads the given JSON file and builds a 2-column matrix of median fitness values.
    Assumes each key is of the form "StructureLabel,0" or "StructureLabel,1".
    Returns:
      - matrix: 2D NumPy array of shape (num_structures, 2)
      - row_labels: sorted list of structure labels (for y-axis)
      - col_names: list of column names for the x-axis (["Few API Calls", "Many API Calls"])
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    cell_values = {}  # Map (structure_label, col_index) -> median fitness
    row_labels_set = set()

    for key, value in data.items():
        if value is None:
            continue
        if "," not in key:
            continue
        parts = key.split(",", 1)
        structure_label = parts[0].strip()
        try:
            col_index = int(parts[1].strip())
        except:
            continue
        # We only expect two columns: 0 and 1
        if col_index not in [0, 1]:
            continue

        fitness_str = value.get("fitness", "")
        median_fit = parse_median_fitness(fitness_str)
        cell_values[(structure_label, col_index)] = median_fit
        row_labels_set.add(structure_label)

    row_labels = sorted(list(row_labels_set))
    col_names = ["Few API Calls", "Many API Calls"]

    num_rows = len(row_labels)
    matrix = np.full((num_rows, 2), np.nan)
    for (row_label, col_index), val in cell_values.items():
        row_idx = row_labels.index(row_label)
        matrix[row_idx, col_index] = val

    return matrix, row_labels, col_names

def plot_heatmap(matrix, row_labels, col_names, output_file):
    """
    Plots a heatmap using the given matrix and labels.
    Uses the "PuBu" colormap.
    Annotates each cell with its value (formatted to one decimal place)
    or "Empty" if the value is NaN.
    Axis titles for structure and API calls are removed.
    """
    num_rows, num_cols = matrix.shape
    plt.figure(figsize=(8, num_rows*0.7 + 2))
    im = plt.imshow(matrix, cmap="PuBu", aspect="auto", origin="upper")
    cbar = plt.colorbar(im)
    cbar.set_label("Median Fitness (%)", rotation=90)

    # Set tick labels (no additional axis titles)
    plt.xticks(ticks=np.arange(num_cols), labels=col_names)
    plt.yticks(ticks=np.arange(num_rows), labels=row_labels)
    
    plt.title("Exemplary MAP after complete run")

    # Annotate each cell
    for i in range(num_rows):
        for j in range(num_cols):
            val = matrix[i, j]
            if np.isnan(val):
                text = "Empty"
            else:
                text = f"{val:.1f}"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format="pdf")
    plt.close()
    print(f"Heatmap saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a heatmap from a final MAP-Elites JSON file in a folder."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the folder containing a single JSON file.")
    parser.add_argument("--output_file", type=str, default="map_elites_heatmap.pdf",
                        help="Filename for the output heatmap (PDF recommended).")
    args = parser.parse_args()

    json_file = load_json_file_from_folder(args.data_dir)
    matrix, row_labels, col_names = build_heatmap_matrix(json_file)
    plot_heatmap(matrix, row_labels, col_names, args.output_file)

if __name__ == "__main__":
    main()