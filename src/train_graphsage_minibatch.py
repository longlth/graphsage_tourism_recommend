import os, time, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

from load_graph_from_csv import load_yelp_graph


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src.long()] * z[dst.long()]).sum(dim=1)


def run_graphsage(epochs=5, save_path="result_graphsage.json"):
    start = time.time()

    # Load từ env.sh
    nodes_csv = os.environ.get("NODES_CSV")
    train_csv = os.environ.get("TRAIN_CSV")
    val_csv   = os.environ.get("VAL_CSV")
    test_csv  = os.environ.get("TEST_CSV")

    data = load_yelp_graph(nodes_csv, train_csv, val_csv, test_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    print("Data loaded on:", device)

    model = GraphSAGE(in_channels=data.x.size(1),
                      hidden_channels=64,
                      out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Mini-batch loaders
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[20, 20],
        batch_size=1024,
        edge_label_index=data.train_pos_edge_index,
        edge_label=torch.ones(data.train_pos_edge_index.size(1)),
        shuffle=True
    )
    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=[20, 20],
        batch_size=1024,
        edge_label_index=data.val_pos_edge_index,
        edge_label=torch.ones(data.val_pos_edge_index.size(1)),
        shuffle=False
    )

    def train():
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model.encode(batch.x, batch.edge_index)

            pos_out = model.decode(z, batch.edge_label_index)
            pos_label = torch.ones(pos_out.size(0), device=device)

            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=pos_out.size(0),
            )
            neg_out = model.decode(z, neg_edge_index)
            neg_label = torch.zeros(neg_out.size(0), device=device)

            out = torch.cat([pos_out, neg_out])
            label = torch.cat([pos_label, neg_label])

            loss = F.binary_cross_entropy_with_logits(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        all_pred, all_label = [], []
        for batch in loader:
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index)

            pos_out = model.decode(z, batch.edge_label_index).sigmoid()
            pos_label = torch.ones(pos_out.size(0), device=device)

            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes,
                num_neg_samples=pos_out.size(0),
            )
            neg_out = model.decode(z, neg_edge_index).sigmoid()
            neg_label = torch.zeros(neg_out.size(0), device=device)

            all_pred.append(torch.cat([pos_out, neg_out]).cpu())
            all_label.append(torch.cat([pos_label, neg_label]).cpu())

        y_pred = torch.cat(all_pred).numpy()
        y_true = torch.cat(all_label).numpy()
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        y_pred_label = (y_pred > 0.5).astype(int)
        precision = precision_score(y_true, y_pred_label)
        recall = recall_score(y_true, y_pred_label)
        f1 = f1_score(y_true, y_pred_label)
        return auc, ap, precision, recall, f1


    best_auc, best_ap = 0, 0
    for epoch in range(1, epochs + 1):
        loss = train()
        auc, ap, precision, recall, f1 = evaluate(val_loader)
        if auc > best_auc:
            best_auc, best_ap, best_precision, best_recall, best_f1 = auc, ap, precision, recall, f1
        print(f"[GraphSAGE] Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {auc:.4f}, Val AP: {ap:.4f}")

    elapsed = time.time() - start
    result = {
        "Model": "GraphSAGE",
        "AUC": best_auc, "AP": best_ap,
        "Precision": best_precision, "Recall": best_recall, "F1": best_f1,
        "Time (s)": elapsed
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[GraphSAGE] Best AUC: {best_auc:.4f}, AP: {best_ap:.4f}, Time: {elapsed:.1f}s")
    print("Saved result to", save_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, nargs="?", default=5, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="result_graphsage_auc.json")
    args = parser.parse_args()

    run_graphsage(epochs=args.epochs, save_path=args.save_path)
