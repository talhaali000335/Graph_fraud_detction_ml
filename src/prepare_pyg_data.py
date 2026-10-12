"""
Graph Fraud Detection AI
Step 7: Prepare PyTorch Geometric Dataset

Converts the NetworkX fraud graph into a PyTorch Geometric
graph suitable for GraphSAGE / GAT.

Target:
    Transaction fraud classification

0 = Normal
1 = Fraud

The graph remains heterogeneous in meaning, but is represented
as one homogeneous PyG graph with node-type and structural
features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx

from torch_geometric.data import Data


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Load graph
# ============================================================

print("\nLoading NetworkX graph...")

graph_file = (
    PROCESSED_DATA_DIR /
    "fraud_graph.graphml"
)

G = nx.read_graphml(
    graph_file
)

print(
    f"NetworkX nodes: {G.number_of_nodes():,}"
)

print(
    f"NetworkX edges: {G.number_of_edges():,}"
)


# ============================================================
# Load transaction data
# ============================================================

transactions = pd.read_csv(
    RAW_DATA_DIR /
    "transactions.csv"
)

print(
    f"Transactions: {len(transactions):,}"
)


# ============================================================
# Create node index mapping
# ============================================================

print("\nCreating node index mapping...")

node_list = list(G.nodes())

node_to_index = {
    node_id: index
    for index, node_id
    in enumerate(node_list)
}

index_to_node = {
    index: node_id
    for node_id, index
    in node_to_index.items()
}


# ============================================================
# Node features
# ============================================================

print("Building node features...")


features = []
labels = []

node_types = []


for node_id in node_list:

    attributes = G.nodes[node_id]

    node_type = attributes.get(
        "node_type",
        "unknown",
    )

    node_types.append(node_type)

    # --------------------------------------------------------
    # Feature 1: Node type
    #
    # customer  = 1
    # sim       = 2
    # device    = 3
    # ip        = 4
    # merchant  = 5
    # transaction = 6
    # --------------------------------------------------------

    node_type_mapping = {
        "customer": 1,
        "sim": 2,
        "device": 3,
        "ip": 4,
        "merchant": 5,
        "transaction": 6,
    }

    type_value = node_type_mapping.get(
        node_type,
        0,
    )

    # --------------------------------------------------------
    # Feature 2: Node degree
    # --------------------------------------------------------

    degree = G.degree(node_id)

    # --------------------------------------------------------
    # Feature 3: Normalized degree
    # --------------------------------------------------------

    normalized_degree = (
        degree /
        max(1, G.number_of_nodes())
    )

    # --------------------------------------------------------
    # Feature 4: Transaction amount
    # --------------------------------------------------------

    amount = float(
        attributes.get(
            "amount",
            0,
        )
    )

    # --------------------------------------------------------
    # Feature 5: Is transaction
    # --------------------------------------------------------

    is_transaction = (
        1
        if node_type == "transaction"
        else 0
    )

    features.append([
        type_value,
        degree,
        normalized_degree,
        amount,
        is_transaction,
    ])

    # --------------------------------------------------------
    # Target label
    #
    # Only transactions have meaningful fraud labels.
    #
    # Non-transaction nodes receive -1.
    # --------------------------------------------------------

    if node_type == "transaction":

        fraud_label = int(
            attributes.get(
                "is_fraud",
                0,
            )
        )

        labels.append(
            fraud_label
        )

    else:

        labels.append(-1)


# ============================================================
# Convert to tensors
# ============================================================

x = torch.tensor(
    features,
    dtype=torch.float,
)

y = torch.tensor(
    labels,
    dtype=torch.long,
)


# ============================================================
# Build edge index
# ============================================================

print("\nBuilding edge index...")

edge_list = []

for source, target in G.edges():

    source_index = node_to_index[
        source
    ]

    target_index = node_to_index[
        target
    ]

    # Add both directions.
    #
    # This makes the graph effectively undirected
    # for message passing.

    edge_list.append([
        source_index,
        target_index,
    ])

    edge_list.append([
        target_index,
        source_index,
    ])


edge_index = torch.tensor(
    edge_list,
    dtype=torch.long,
).t().contiguous()


# ============================================================
# Identify transaction nodes
# ============================================================

transaction_indices = [
    index
    for index, node_type
    in enumerate(node_types)
    if node_type == "transaction"
]


transaction_indices = torch.tensor(
    transaction_indices,
    dtype=torch.long,
)


print(
    f"Transaction nodes: "
    f"{len(transaction_indices):,}"
)


# ============================================================
# Create train / validation / test masks
# ============================================================

print("\nCreating train/validation/test masks...")


# Only transactions are prediction targets.
transaction_labels = y[
    transaction_indices
]


# Fraud and normal indices
fraud_indices = transaction_indices[
    transaction_labels == 1
]

normal_indices = transaction_indices[
    transaction_labels == 0
]


# Shuffle
generator = torch.Generator().manual_seed(
    SEED
)

fraud_indices = fraud_indices[
    torch.randperm(
        len(fraud_indices),
        generator=generator,
    )
]

normal_indices = normal_indices[
    torch.randperm(
        len(normal_indices),
        generator=generator,
    )
]


# ============================================================
# Stratified split
# ============================================================

def split_indices(indices):

    n = len(indices)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    train = indices[
        :train_end
    ]

    validation = indices[
        train_end:val_end
    ]

    test = indices[
        val_end:
    ]

    return train, validation, test


fraud_train, fraud_val, fraud_test = (
    split_indices(fraud_indices)
)

normal_train, normal_val, normal_test = (
    split_indices(normal_indices)
)


train_indices = torch.cat([
    fraud_train,
    normal_train,
])

val_indices = torch.cat([
    fraud_val,
    normal_val,
])

test_indices = torch.cat([
    fraud_test,
    normal_test,
])


# ============================================================
# Masks
# ============================================================

train_mask = torch.zeros(
    G.number_of_nodes(),
    dtype=torch.bool,
)

val_mask = torch.zeros(
    G.number_of_nodes(),
    dtype=torch.bool,
)

test_mask = torch.zeros(
    G.number_of_nodes(),
    dtype=torch.bool,
)


train_mask[
    train_indices
] = True

val_mask[
    val_indices
] = True

test_mask[
    test_indices
] = True


# ============================================================
# Create PyG Data object
# ============================================================

print("\nCreating PyTorch Geometric Data object...")


data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)


# Store additional information
data.transaction_mask = (
    torch.tensor(
        [
            node_type == "transaction"
            for node_type in node_types
        ],
        dtype=torch.bool,
    )
)


# ============================================================
# Display information
# ============================================================

print("\n" + "=" * 60)
print("PYTORCH GEOMETRIC DATASET")
print("=" * 60)

print(
    f"\nNodes: "
    f"{data.num_nodes:,}"
)

print(
    f"Edges: "
    f"{data.num_edges:,}"
)

print(
    f"Features per node: "
    f"{data.num_node_features}"
)

print(
    f"Transaction nodes: "
    f"{data.transaction_mask.sum().item():,}"
)

print(
    f"\nTraining nodes: "
    f"{data.train_mask.sum().item():,}"
)

print(
    f"Validation nodes: "
    f"{data.val_mask.sum().item():,}"
)

print(
    f"Test nodes: "
    f"{data.test_mask.sum().item():,}"
)


# ============================================================
# Fraud distribution
# ============================================================

print("\nTraining fraud distribution:")

train_labels = data.y[
    data.train_mask
]

train_fraud = (
    train_labels == 1
).sum().item()

train_normal = (
    train_labels == 0
).sum().item()


print(
    f"  Normal: "
    f"{train_normal:,}"
)

print(
    f"  Fraud:  "
    f"{train_fraud:,}"
)


print("\nValidation fraud distribution:")

val_labels = data.y[
    data.val_mask
]

print(
    f"  Normal: "
    f"{(val_labels == 0).sum().item():,}"
)

print(
    f"  Fraud:  "
    f"{(val_labels == 1).sum().item():,}"
)


print("\nTest fraud distribution:")

test_labels = data.y[
    data.test_mask
]

print(
    f"  Normal: "
    f"{(test_labels == 0).sum().item():,}"
)

print(
    f"  Fraud:  "
    f"{(test_labels == 1).sum().item():,}"
)


# ============================================================
# Save
# ============================================================

output_file = (
    PROCESSED_DATA_DIR /
    "fraud_graph_pyg.pt"
)

torch.save(
    data,
    output_file,
)


# Save node mapping
mapping_df = pd.DataFrame({
    "node_index": range(
        len(node_list)
    ),
    "node_id": node_list,
    "node_type": node_types,
})


mapping_file = (
    PROCESSED_DATA_DIR /
    "node_mapping.csv"
)

mapping_df.to_csv(
    mapping_file,
    index=False,
)


# ============================================================
# Complete
# ============================================================

print("\nSaved PyG dataset:")

print(output_file)

print("\nSaved node mapping:")

print(mapping_file)

print("\nStep 7 complete.")

print(
    "\nNext step:"
)

print(
    "Train the first GraphSAGE fraud "
    "detection model."
)

print("=" * 60)