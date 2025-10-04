# Tourism Place Recommendation using Graph Neural Networks

This project implements and compares various Graph Neural Network (GNN) approaches for tourism place recommendation using the Yelp dataset. The system uses link prediction to recommend places to users based on their historical interactions and similarities.

## Models Implemented

- **GraphSAGE**: Implementation of the GraphSAGE algorithm with both full-batch and mini-batch training
- **GCN (Graph Convolutional Network)**: Basic GCN implementation for link prediction
- **Node2Vec**: Graph embedding approach for comparison
- **Baseline Heuristics**: Simple graph-based heuristics (Jaccard Coefficient) for baseline comparison

## Project Structure

```
training_graphsage/
├── data/                     # Data directory
│   ├── edges_test_subset/    # Test set edges
│   ├── edges_train_subset/   # Training set edges
│   ├── edges_val_subset/     # Validation set edges
│   └── nodes_subset/         # Node features and metadata
├── src/                      # Source code
│   ├── baseline_heuristics.py    # Baseline model implementation
│   ├── benchmark_runner.py       # Benchmarking and visualization
│   ├── load_graph_from_csv.py    # Data loading utilities
│   ├── train_gcn.py             # GCN model implementation
│   ├── train_graphsage_fullbatch.py  # Full-batch GraphSAGE
│   ├── train_graphsage_minibatch.py  # Mini-batch GraphSAGE
│   ├── train_node2vec.py        # Node2Vec implementation
│   └── utils_eval.py            # Evaluation utilities
├── result/                  # Results and visualizations
├── requirements.txt         # Python dependencies
├── env.sh                   # Environment variables
└── start.sh                # Training pipeline script
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/longlth/graphsage_tourism_recommend.git
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
source env.sh
```

## Running the Project

To run the complete training and evaluation pipeline:

```bash
./start.sh
```

This will:
1. Run baseline heuristics
2. Train Node2Vec model
3. Train GCN model
4. Train GraphSAGE model
5. Generate comparison visualizations

## Visualizations and Results

The benchmark results are saved in the `result/` directory:
- `result/visualizations/model_comparison_bar.png`: Bar plot comparing models across metrics
- `result/visualizations/model_comparison_radar.png`: Radar chart showing model balance
- `result/visualizations/model_comparison_heatmap.png`: Heatmap of model performance
- `result/visualizations/auc_ap_time_scatter.png`: Scatter plot of AUC vs AP with execution time
- `result/benchmark_results.csv`: Detailed numerical results

## Model Performance Metrics

The following metrics are used for evaluation:
- AUC (Area Under the ROC Curve)
- AP (Average Precision)
- Precision
- Recall
- F1 Score
- Execution Time

## Requirements

- Python 3.8+
- PyTorch 2.8.0
- PyTorch Geometric 2.6.1
- Other dependencies listed in requirements.txt

## Configuration

Key hyperparameters and settings can be modified in:
- `env.sh`: Data paths and default training parameters
- Model-specific parameters in respective training files
