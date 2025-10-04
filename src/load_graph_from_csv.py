import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def load_yelp_graph(nodes_csv, train_csv, val_csv, test_csv):
    nodes_df = pd.read_csv(nodes_csv, quotechar='"', escapechar='\\')
    train_df = pd.read_csv(train_csv, quotechar='"', escapechar='\\')
    val_df   = pd.read_csv(val_csv, quotechar='"', escapechar='\\')
    test_df  = pd.read_csv(test_csv, quotechar='"', escapechar='\\')

    # Build mapping id -> idx
    nodes_df['id'] = nodes_df['id'].astype(str)
    all_ids = pd.Index(nodes_df['id'].unique())
    id2idx = {nid: i for i, nid in enumerate(all_ids)}
    num_nodes = len(id2idx)

    # Build features (user/business one-hot)
    X = np.zeros((num_nodes, 2), dtype=np.float32)
    for _, row in nodes_df.iterrows():
        i = id2idx.get(str(row['id']))
        if i is None:
            continue
        if str(row['type']).lower() == 'user':
            X[i, 0] = 1.0
        else:
            X[i, 1] = 1.0
    x = torch.tensor(X, dtype=torch.float)

    # Build edges
    def map_edges(df, name="train"):
        src = df['src'].astype(str).map(id2idx)
        dst = df['dst'].astype(str).map(id2idx)
        mask = src.notna() & dst.notna()
        dropped = (~mask).sum()
        if dropped > 0:
            print(f"Warning: dropped {dropped} edges in {name} (id not in nodes)")
        src, dst = src[mask].astype(int).values, dst[mask].astype(int).values
        if len(src) == 0:
            return torch.empty((2,0), dtype=torch.long)
        edges = np.vstack([src, dst]).astype(np.int64)
        return torch.from_numpy(edges)

    edge_index_train = map_edges(train_df)
    edge_index_val   = map_edges(val_df)
    edge_index_test  = map_edges(test_df)

    edge_index_train = to_undirected(edge_index_train, num_nodes=num_nodes)

    # Create Data object
    data = Data(x=x, edge_index=edge_index_train, num_nodes=num_nodes)
    data.train_pos_edge_index = edge_index_train
    data.val_pos_edge_index   = edge_index_val
    data.test_pos_edge_index  = edge_index_test

    return data

if __name__ == "__main__":
    nodes_csv = os.environ.get("NODES_CSV")
    train_csv = os.environ.get("TRAIN_CSV")
    val_csv   = os.environ.get("VAL_CSV")
    test_csv  = os.environ.get("TEST_CSV")

    data = load_yelp_graph(
        nodes_csv,
        train_csv,
        val_csv,
        test_csv,
    )
    print(data)
