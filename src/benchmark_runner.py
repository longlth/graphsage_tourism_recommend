import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

RESULT_DIR = "result"

def load_results(result_dir=RESULT_DIR):
    results = []
    for fname in os.listdir(result_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(result_dir, fname)
            with open(fpath, "r") as f:
                data = json.load(f)
                results.append(data)
    return results

def plot_bar_comparison(df, metrics, save_path):
    """Create a grouped bar plot comparing models across metrics."""
    bar_width = 0.15
    index = np.arange(len(df))

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        plt.bar(index + i*bar_width, df[metric], bar_width, label=metric)

    plt.xticks(index + bar_width*(len(metrics)/2), df["Model"], rotation=45)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_radar_chart(df, metrics, save_path):
    """Create a radar chart comparing models across metrics."""
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for idx, model in enumerate(df["Model"]):
        values = df[df["Model"] == model][metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model, marker='o')
        ax.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], metrics)
    ax.set_title("Model Performance Radar Chart")
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(df, metrics, save_path):
    """Create a heatmap showing model performance across metrics."""
    plt.figure(figsize=(10, 8))
    pivot_df = df.pivot_table(index="Model", values=metrics)
    sns.heatmap(pivot_df, annot=True, cmap="YlOrRd", fmt=".3f", cbar_kws={'label': 'Score'})
    plt.title("Performance Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_performance_scatter(df, save_path):
    """Create a scatter plot comparing AUC vs AP with execution time as point size."""
    plt.figure(figsize=(10, 8))
    plt.scatter(df["AUC"], df["AP"], s=df["Time (s)"]*100, alpha=0.6)
    
    for i, model in enumerate(df["Model"]):
        plt.annotate(model, 
                    (df["AUC"].iloc[i], df["AP"].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("AUC Score")
    plt.ylabel("AP Score")
    plt.title("AUC vs AP Performance (point size = execution time)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_results_directory(base_dir):
    """Create visualization directory if it doesn't exist."""
    vis_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def main():
    results = load_results()
    if not results:
        print("No result JSON files found in", RESULT_DIR)
        return

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, "benchmark_results.csv"), index=False)
    print("=== Benchmark Results ===")
    print(df)

    metrics = ["AUC", "AP", "Precision", "Recall", "F1"]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("No evaluation metrics found in results JSON.")
        return

    # Create visualization directory
    vis_dir = create_results_directory(RESULT_DIR)

    # Generate various visualizations
    plot_bar_comparison(df, available_metrics, 
                       os.path.join(vis_dir, "model_comparison_bar.png"))
    plot_radar_chart(df, available_metrics, 
                    os.path.join(vis_dir, "model_comparison_radar.png"))
    plot_heatmap(df, available_metrics, 
                os.path.join(vis_dir, "model_comparison_heatmap.png"))
    plot_performance_scatter(df, 
                           os.path.join(vis_dir, "auc_ap_time_scatter.png"))

    # Print summary statistics
    print("\n=== Performance Summary ===")
    print("\nBest Model by Metric:")
    for metric in available_metrics:
        best_model = df.loc[df[metric].idxmax()]
        print(f"{metric}: {best_model['Model']} ({best_model[metric]:.4f})")

    print("\nExecution Time Comparison:")
    time_summary = df[["Model", "Time (s)"]].sort_values("Time (s)")
    print(time_summary.to_string(index=False))

if __name__ == "__main__":
    main()
