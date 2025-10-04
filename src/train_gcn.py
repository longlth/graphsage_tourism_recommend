import os, argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

from load_graph_from_csv import load_yelp_graph
from utils_eval import evaluate_link_prediction


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

def run_gcn(epochs=5, save_path="result_gcn.json"):
    start = time.time()

    nodes_csv = os.environ.get("NODES_CSV")
    train_csv = os.environ.get("TRAIN_CSV")
    val_csv   = os.environ.get("VAL_CSV")
    test_csv  = os.environ.get("TEST_CSV")

    data = load_yelp_graph(nodes_csv, train_csv, val_csv, test_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GCN(data.x.size(1), 64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_metrics = {"AUC": 0, "AP": 0, "Precision": 0, "Recall": 0, "F1": 0}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)

        pos_out = model.decode(z, data.train_pos_edge_index)
        pos_label = torch.ones(pos_out.size(0), device=device)

        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_out.size(0),
        )
        neg_out = model.decode(z, neg_edge_index)
        neg_label = torch.zeros(neg_out.size(0), device=device)

        out = torch.cat([pos_out, neg_out])
        label = torch.cat([pos_label, neg_label])

        loss = F.binary_cross_entropy_with_logits(out, label)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index)
            pos_out = model.decode(z, data.val_pos_edge_index).sigmoid()
            pos_label = torch.ones(pos_out.size(0), device=device)

            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_out.size(0),
            )
            neg_out = model.decode(z, neg_edge_index).sigmoid()
            neg_label = torch.zeros(neg_out.size(0), device=device)

            y_pred = torch.cat([pos_out, neg_out]).cpu().numpy()
            y_true = torch.cat([pos_label, neg_label]).cpu().numpy()

            auc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            y_pred_label = (y_pred > 0.5).astype(int)
            precision = precision_score(y_true, y_pred_label)
            recall = recall_score(y_true, y_pred_label)
            f1 = f1_score(y_true, y_pred_label)

        if auc > best_metrics["AUC"]:  # lưu best theo AUC
            best_metrics = {
                "AUC": auc, "AP": ap,
                "Precision": precision, "Recall": recall, "F1": f1
            }

        print(f"[GCN] Epoch {epoch}, Loss: {loss.item():.4f}, "
              f"AUC: {auc:.4f}, AP: {ap:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    elapsed = time.time() - start
    best_metrics["Model"] = "GCN"
    best_metrics["Time (s)"] = elapsed

    with open(save_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"[GCN] Best metrics: {best_metrics}")
    return best_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, nargs="?", default=5, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="result_gcn.json")
    args = parser.parse_args()

    run_gcn(epochs=args.epochs, save_path=args.save_path)
