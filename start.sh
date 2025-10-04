#!/bin/bash
set -e

source env.sh

echo ">>> Running baseline heuristics..."
python src/baseline_heuristics.py --sample_size 5000 --save_path result/result_heuristics.json

echo ">>> Running Node2Vec..."
python src/train_node2vec.py $DEFAULT_EPOCHS --save_path result/result_node2vec.json

echo ">>> Running GCN..."
python src/train_gcn.py $DEFAULT_EPOCHS --save_path result/result_gcn.json

echo ">>> Running GraphSAGE..."
python src/train_graphsage_minibatch.py $DEFAULT_EPOCHS --save_path result/result_graphsage_auc.json

echo ">>> Aggregating results..."
python src/benchmark_runner.py

# echo ">>> Visualizing results..."
# python src/better_visualize_comparison.py
