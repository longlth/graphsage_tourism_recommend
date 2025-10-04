# train_graphsage_fullbatch.py
import os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from load_graph_from_csv import load_yelp_graph
from sklearn.metrics import roc_auc_score, average_precision_score

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

def run_graphsage_fullbatch(epochs=5, save_path="result_graphsage_full.json"):
    start = time.time()

    nodes_csv = os.environ.get("NODES_CSV")
    train_csv = os.environ.get("TRAIN_CSV")
    val_csv   = os.environ.get("VAL_CSV")
    test_csv  = os.environ.get("TEST_CSV")

    data = load_yelp_graph(nodes_csv, train_csv, val_csv, test_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=128,
        out_channels=64
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)

        pos_out = model.decode(z, data.train_pos_edge_index)
        pos_label = torch.ones(pos_out.size(0), device=device)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
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
        return loss.item()

    @torch.no_grad()
    def test(pos_edge_index):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        pos_out = model.decode(z, pos_edge_index).sigmoid()
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_out.size(0),
        )
        neg_out = model.decode(z, neg_edge_index).sigmoid()

        y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))])
        y_pred = torch.cat([pos_out, neg_out])
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
        ap = average_precision_score(y_true.cpu(), y_pred.cpu())
        return auc, ap

    for epoch in range(1, epochs + 1):
        loss = train()
        val_auc, val_ap = test(data.val_pos_edge_index)
        print(f"[Epoch {epoch}] Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")

    test_auc, test_ap = test(data.test_pos_edge_index)
    elapsed = time.time() - start

    result = {"Model": "GraphSAGE-FullBatch", "AUC": test_auc, "AP": test_ap, "Time (s)": elapsed}
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[GraphSAGE Full] Test AUC: {test_auc:.4f}, AP: {test_ap:.4f}, Time: {elapsed:.1f}s")
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, nargs="?", default=5, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="result_graphsage_full.json")
    args = parser.parse_args()
    run_graphsage_fullbatch(epochs=args.epochs, save_path=args.save_path)
