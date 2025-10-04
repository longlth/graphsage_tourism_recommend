import os, argparse, json, time, random
import networkx as nx
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

def run_heuristics(sample_size=2000, save_path="../result/result_heuristics.json"):
    start = time.time()

    train_csv = os.environ.get("TRAIN_CSV")
    test_csv  = os.environ.get("TEST_CSV")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # Build graph
    G = nx.Graph()
    for _, row in train_df.iterrows():
        G.add_edge(row["src"], row["dst"])

    preds, labels = [], []

    # Positive edges (from test set)
    for _, row in test_df.sample(sample_size).iterrows():
        u, v = row["src"], row["dst"]
        try:
            score = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
        except:
            score = 0.0
        preds.append(score)
        labels.append(1)

    # Negative edges (random pairs)
    nodes = list(G.nodes())
    for _ in range(sample_size):
        u, v = random.sample(nodes, 2)
        try:
            score = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
        except:
            score = 0.0
        preds.append(score)
        labels.append(0)

    # Convert to numpy
    y_true = labels
    y_pred = preds
    y_pred_label = [1 if p > 0.5 else 0 for p in preds]

    # Metrics
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    elapsed = time.time() - start

    result = {
        "Model": "Jaccard",
        "AUC": auc, "AP": ap,
        "Precision": precision, "Recall": recall, "F1": f1,
        "Time (s)": elapsed
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Jaccard] AUC={auc:.4f}, AP={ap:.4f}, "
          f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
          f"Time={elapsed:.1f}s")
    print("Saved result to", save_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=2000, help="Number of edges to sample")
    parser.add_argument("--save_path", type=str, default="result_heuristics.json")
    args = parser.parse_args()

    run_heuristics(sample_size=args.sample_size, save_path=args.save_path)
