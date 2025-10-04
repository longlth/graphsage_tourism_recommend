import os, argparse, json, time
import torch
from torch_geometric.nn.models import Node2Vec
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from torch_geometric.utils import negative_sampling

from load_graph_from_csv import load_yelp_graph


def run_node2vec(epochs=5, save_path="result_node2vec.json"):
    start = time.time()

    nodes_csv = os.environ.get("NODES_CSV")
    train_csv = os.environ.get("TRAIN_CSV")
    val_csv   = os.environ.get("VAL_CSV")
    test_csv  = os.environ.get("TEST_CSV")

    data = load_yelp_graph(nodes_csv, train_csv, val_csv, test_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = Node2Vec(
        data.edge_index, embedding_dim=64, walk_length=20, context_size=10,
        walks_per_node=5, num_negative_samples=1, sparse=True
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Node2Vec] Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

    # Embeddings
    emb = model.forward().cpu()

    # Link prediction eval
    pos_edge = data.test_pos_edge_index
    neg_edge = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=pos_edge.size(1)
    )

    pos_score = (emb[pos_edge[0]] * emb[pos_edge[1]]).sum(dim=1).sigmoid()
    neg_score = (emb[neg_edge[0]] * emb[neg_edge[1]]).sum(dim=1).sigmoid()

    y_true = torch.cat([
        torch.ones(pos_score.size(0)),
        torch.zeros(neg_score.size(0))
    ]).cpu().numpy()

    y_pred = torch.cat([pos_score, neg_score]).detach().cpu().numpy()

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    y_pred_label = (y_pred > 0.5).astype(int)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    elapsed = time.time() - start

    result = {
        "Model": "Node2Vec",
        "AUC": auc,
        "AP": ap,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Time (s)": elapsed
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Node2Vec] Test AUC: {auc:.4f}, AP: {ap:.4f}, Time: {elapsed:.1f}s")
    print("Saved result to", save_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, nargs="?", default=5, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="result_node2vec.json")
    args = parser.parse_args()

    run_node2vec(epochs=args.epochs, save_path=args.save_path)
