import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score


@torch.no_grad()
def evaluate_link_prediction(model, data, edge_index, device="cpu"):
    """
    Evaluate link prediction (AUC, AP) on given edge_index.
    """
    model.eval()
    data = data.to(device)
    z = model.encode(data.x, data.edge_index)

    # Positive edges
    pos_out = model.decode(z, edge_index).sigmoid().cpu()
    pos_label = torch.ones(pos_out.size(0))

    # Negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_out.size(0),
    )
    neg_out = model.decode(z, neg_edge_index).sigmoid().cpu()
    neg_label = torch.zeros(neg_out.size(0))

    out = torch.cat([pos_out, neg_out])
    label = torch.cat([pos_label, neg_label])

    auc = roc_auc_score(label, out)
    ap = average_precision_score(label, out)
    return auc, ap
