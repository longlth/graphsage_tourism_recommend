#!/bin/bash
# env.sh

export NODES_CSV="data/nodes_subset/nodes_subset.csv"
export TRAIN_CSV="data/edges_train_subset/edges_train_subset.csv"
export VAL_CSV="data/edges_val_subset/edges_val_subset.csv"
export TEST_CSV="data/edges_test_subset/edges_test_subset.csv"

export DEFAULT_EPOCHS=5
export DEFAULT_BATCH_SIZE=1024
