import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def find_fitness(string):
    '''Returns a dictionary with the confidence level, lower bound, upper bound, and median of the fitness string'''
    confidence_match = re.search(r'(\d+)% Bootstrap Confidence Interval', string)
    confidence_level = float(confidence_match.group(1)) if confidence_match else None
    
    interval_match = re.search(r'\((\d+\.\d+)%, (\d+\.\d+)%\)', string)
    lower_bound = float(interval_match.group(1)) if interval_match else None
    upper_bound = float(interval_match.group(2)) if interval_match else None

    median_match = re.search(r'Median: (\d+\.\d+)%', string)
    median = float(median_match.group(1)) if median_match else None
    
    return {
        'confidence_level': confidence_level,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'median': median
    }
    
def get_results(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
    try:
        return [
            {
                "name": result["name"],
                "generation": result["generation"],
                "fitness": find_fitness(result["fitness"]),
                "test_fitness": find_fitness(result.get("test_fitness", "-1% Bootstrap Confidence Interval: (-1.0%, -1.0%), Median: -1.0%"))
            }
            for result in results
        ]
    except:
        print(results_file)
        assert False

def plot_agents_per_epoch(results, string, save_dir):
    print(string, save_dir)
    domain = string.split("_")[-1]
    epochs = [result["generation"] for result in results]
    fitnesses = [result["fitness"]["median"] for result in results] 
    averages = [sum(fitnesses[:i+1])/(i+1) for i in range(len(fitnesses))]
    tops = [max(fitnesses[:i+1]) for i in range(len(fitnesses))]
    plt.plot(epochs, fitnesses, alpha=0.5)
    plt.plot(epochs, averages)
    plt.plot(epochs, tops)
    plt.xlabel("Epoch")
    plt.ylabel("Fitness")
    plt.legend(["Individual Fitness", "Running Average", "Top Fitness"])
    os.makedirs(save_dir, exist_ok=True)
    plt.title(domain)
    plt.savefig(f"{save_dir}/{string}_fitness_per_epoch.png")
    plt.close()




def iterate_and_apply(directory, func, regex, string_on_plots, save_dir):
    results_dict = {}
    for file in os.listdir(directory):
        if re.match(regex, file):
            domain = file.split("_")[0]
            results_dict[domain] = []
            results = get_results(os.path.join(directory, file))
            results_dict[domain].extend(results)
            func(results, string_on_plots+'_'+domain, save_dir)
    return results_dict

def bar_comparison(baseline_results_list, og_results, string_on_plots, save_dir):
    domains = np.unique(list(og_results.keys()))
    domain_results_for_plots = {str(d): [[] for _ in range(len(baseline_results_list)*2+2)] for d in domains}
    
    for domain, results in og_results.items():
        for result in results:
            domain_results_for_plots[domain][0].append(result['generation'])
            domain_results_for_plots[domain][1].append(result['fitness']['median'])
    c = 2
    for baseline_results in baseline_results_list:
        for domain, results in baseline_results.items():
            for result in results:
                domain_results_for_plots[domain][c].append(result['generation'])
                domain_results_for_plots[domain][c+1].append(result['fitness']['median'])
        c += 2
    
    print(domain_results_for_plots)

    for domain, results in domain_results_for_plots.items():
        plt.figure(figsize=(10, 6))
        
        # Calculate max-so-far for OG results (index 1)
        fitnesses_og = results[1]
        max_so_far_og = [max(fitnesses_og[:i+1]) for i in range(len(fitnesses_og))]
        print(max_so_far_og)
        # Create a list to store max-so-far values for baseline runs only
        baseline_max_values = []
        x_values = [results[0]]  # Store x-axis values
        x_values = ['initial' if x == 'initial' else x for x in x_values]
        
        # Process baseline results (starting from index 2)
        c = 2
        while c < len(results):
            if len(results[c]) > 0 and len(results[c+1]) > 0:  # Check if there are values
                fitnesses = results[c+1]
                max_so_far = [max(fitnesses[:i+1]) for i in range(len(fitnesses))]
                baseline_max_values.append(max_so_far)
                x_values.append(results[c])
            c += 2
        
        # Find common x-axis points for comparison
        common_x = results[0]
        
        # Convert x-axis to numeric values for plotting
        numeric_x = []
        x_tick_positions = []
        x_tick_labels = []
        
        for i, x in enumerate(common_x):
            if x == 'initial':
                numeric_x.append(-len([v for v in common_x if v == 'initial']) + i)
                x_tick_positions.append(numeric_x[-1])
                x_tick_labels.append('init')
            else:
                numeric_x.append(int(x))
                if i % 5 == 0 or i == len(common_x)-1:
                    x_tick_positions.append(numeric_x[-1])
                    x_tick_labels.append(str(x))
        
        # For each x point, find min and max across BASELINE runs only
        min_values = []
        max_values = []
        
        for i, x in enumerate(common_x):
            current_values = [vals[i] if i < len(vals) else None for vals in baseline_max_values]
            current_values = [v for v in current_values if v is not None]
            
            if current_values:
                min_values.append(min(current_values))
                max_values.append(max(current_values))
            else:
                # If no baseline values available for this point, use None
                min_values.append(None)
                max_values.append(None)
        
        # Remove None values for fill_between (can't fill None regions)
        valid_indices = [i for i, v in enumerate(min_values) if v is not None]
        if valid_indices:
            valid_x = [numeric_x[i] for i in valid_indices]
            valid_min = [min_values[i] for i in valid_indices]
            valid_max = [max_values[i] for i in valid_indices]
            
            # Fill the area between baselines
            plt.fill_between(valid_x, valid_min, valid_max, alpha=0.3, color='red', label='Baseline Range (gpt-4o-mini and gemini-1.5-pro)')
        
        # Plot the min and max lines for baselines
        #if all(v is not None for v in min_values):
        plt.plot(numeric_x, min_values, 'r--')#, label='Upper bound on worst run')
        #if all(v is not None for v in max_values):
        plt.plot(numeric_x, max_values, 'r--')#, label='Upper bound on best run')
        
        # Plot the OG run separately with a distinct color and line style
        plt.plot(numeric_x, max_so_far_og, 'g-', linewidth=2, label='Original Run (gpt-4o-mini and gpt-3.5-turbo)')
        
        # Set custom x-ticks
        plt.xticks(x_tick_positions, x_tick_labels)
        
        plt.xlabel("Generation")
        plt.ylabel("Fitness (%)")
        plt.title(f"Performance Range for {domain}")
        plt.legend()
        
        # Save the figure
        #os.makedirs(save_dir, exist_ok=True)
        #plt.savefig(f"{save_dir}/{domain}_performance_range.png")
        plt.show()
        plt.close()



def main():
    analis_folder = "analysis"
    baseline_results_dir = "results/google_gemini_tests"
    og_paper_results_dir = "results/og"

    #Plot baseline results
    print('Plotting OG')
    og_results = iterate_and_apply(og_paper_results_dir, plot_agents_per_epoch, r".*gpt3\.5.*", "og", analis_folder+"/og")
    print('Plotting baseline_run_1')
    baseline_results = iterate_and_apply(baseline_results_dir, plot_agents_per_epoch, r".*run_final(?!.*_2).*", "baseline_run_1", analis_folder+"/baselinev1")

    print('Plotting baseline_run_2')
    baseline_results_2 = iterate_and_apply(baseline_results_dir, plot_agents_per_epoch, r".*run_final_2.*", "baseline_run_2", analis_folder+"/baselinev2")
    
    bar_comparison([baseline_results, baseline_results_2], og_results, "baseline_run_1", analis_folder+"/baselinev1")
if __name__ == "__main__":
    main()

