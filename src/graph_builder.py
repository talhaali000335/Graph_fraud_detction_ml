"""
Graph Fraud Detection AI
Graph Builder

Converts the synthetic fraud datasets into a heterogeneous
NetworkX graph.

Node types:
    Customer
    SIM
    Device
    IP
    Transaction
    Merchant

Edge types:
    Customer -> SIM
    Customer -> Device
    Customer -> IP
    Customer -> Transaction
    Transaction -> Merchant
    Transaction -> Device
    Transaction -> IP

The graph is saved as a NetworkX GraphML file and also as
a simple edge list for inspection.
"""

from pathlib import Path

import pandas as pd
import networkx as nx


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
# Load datasets
# ============================================================

print("\nLoading datasets...")

customers = pd.read_csv(
    RAW_DATA_DIR / "customers.csv"
)

sims = pd.read_csv(
    RAW_DATA_DIR / "sims.csv"
)

devices = pd.read_csv(
    RAW_DATA_DIR / "devices.csv"
)

ips = pd.read_csv(
    RAW_DATA_DIR / "ips.csv"
)

merchants = pd.read_csv(
    RAW_DATA_DIR / "merchants.csv"
)

transactions = pd.read_csv(
    RAW_DATA_DIR / "transactions.csv"
)


print("Datasets loaded successfully.")


# ============================================================
# Create graph
# ============================================================

print("\nCreating graph...")

G = nx.Graph()


# ============================================================
# Add CUSTOMER nodes
# ============================================================

print("Adding customer nodes...")

for _, row in customers.iterrows():

    G.add_node(
        row["customer_id"],
        node_type="customer",
        age=int(row["age"]),
        nationality=row["nationality"],
        occupation=row["occupation"],
        account_age_days=int(
            row["account_age_days"]
        ),
    )


# ============================================================
# Add SIM nodes
# ============================================================

print("Adding SIM nodes...")

for _, row in sims.iterrows():

    sim_id = row["sim_id"]

    G.add_node(
        sim_id,
        node_type="sim",
        msisdn=str(row["msisdn"]),
    )


# ============================================================
# Add DEVICE nodes
# ============================================================

print("Adding device nodes...")

for _, row in devices.iterrows():

    device_id = row["device_id"]

    G.add_node(
        device_id,
        node_type="device",
        brand=row["brand"],
        imei=str(row["imei"]),
    )


# ============================================================
# Add IP nodes
# ============================================================

print("Adding IP nodes...")

for _, row in ips.iterrows():

    ip_id = row["ip_id"]

    G.add_node(
        ip_id,
        node_type="ip",
        ip_address=row["ip_address"],
        country=row["country"],
    )


# ============================================================
# Add MERCHANT nodes
# ============================================================

print("Adding merchant nodes...")

for _, row in merchants.iterrows():

    merchant_id = row["merchant_id"]

    G.add_node(
        merchant_id,
        node_type="merchant",
        category=row["category"],
    )


# ============================================================
# Add TRANSACTION nodes
# ============================================================

print("Adding transaction nodes...")

for _, row in transactions.iterrows():

    transaction_id = row["transaction_id"]

    G.add_node(
        transaction_id,
        node_type="transaction",
        amount=float(row["amount"]),
        is_fraud=int(row["is_fraud"]),
        fraud_type=row["fraud_type"],
    )


# ============================================================
# Customer relationships
# ============================================================

print("Creating customer relationships...")

for _, row in sims.iterrows():

    customer_id = row["customer_id"]
    sim_id = row["sim_id"]

    G.add_edge(
        customer_id,
        sim_id,
        edge_type="owns_sim",
    )


# ============================================================
# Transaction relationships
# ============================================================

print("Creating transaction relationships...")

for _, row in transactions.iterrows():

    transaction_id = row["transaction_id"]

    customer_id = row["customer_id"]
    sim_id = row["sim_id"]
    device_id = row["device_id"]
    ip_id = row["ip_id"]
    merchant_id = row["merchant_id"]

    # Customer -> Transaction
    G.add_edge(
        customer_id,
        transaction_id,
        edge_type="made_transaction",
    )

    # SIM -> Transaction
    G.add_edge(
        sim_id,
        transaction_id,
        edge_type="used_sim",
    )

    # Device -> Transaction
    G.add_edge(
        device_id,
        transaction_id,
        edge_type="used_device",
    )

    # IP -> Transaction
    G.add_edge(
        ip_id,
        transaction_id,
        edge_type="used_ip",
    )

    # Merchant -> Transaction
    G.add_edge(
        merchant_id,
        transaction_id,
        edge_type="merchant_transaction",
    )


# ============================================================
# Save graph
# ============================================================

print("\nSaving graph...")

graph_file = (
    PROCESSED_DATA_DIR /
    "fraud_graph.graphml"
)

nx.write_graphml(
    G,
    graph_file,
)


# ============================================================
# Save edge list
# ============================================================

edge_file = (
    PROCESSED_DATA_DIR /
    "fraud_graph_edges.csv"
)

edges = []

for source, target, attributes in G.edges(
    data=True
):

    edges.append({
        "source": source,
        "target": target,
        "edge_type": attributes.get(
            "edge_type",
            "unknown",
        ),
    })


edges_df = pd.DataFrame(edges)

edges_df.to_csv(
    edge_file,
    index=False,
)


# ============================================================
# Statistics
# ============================================================

print("\n" + "=" * 60)
print("FRAUD GRAPH CREATED")
print("=" * 60)

print(
    f"\nTotal nodes: {G.number_of_nodes():,}"
)

print(
    f"Total edges: {G.number_of_edges():,}"
)


# ============================================================
# Node statistics
# ============================================================

node_types = {}

for _, attributes in G.nodes(
    data=True
):

    node_type = attributes.get(
        "node_type",
        "unknown",
    )

    node_types[node_type] = (
        node_types.get(node_type, 0) + 1
    )


print("\nNodes by type:")

for node_type, count in sorted(
    node_types.items()
):

    print(
        f"  {node_type:<15} {count:,}"
    )


# ============================================================
# Edge statistics
# ============================================================

edge_types = {}

for _, _, attributes in G.edges(
    data=True
):

    edge_type = attributes.get(
        "edge_type",
        "unknown",
    )

    edge_types[edge_type] = (
        edge_types.get(edge_type, 0) + 1
    )


print("\nEdges by type:")

for edge_type, count in sorted(
    edge_types.items()
):

    print(
        f"  {edge_type:<25} {count:,}"
    )


# ============================================================
# Fraud transaction statistics
# ============================================================

fraud_transactions = sum(
    1
    for node, attributes
    in G.nodes(data=True)
    if (
        attributes.get("node_type")
        == "transaction"
        and attributes.get("is_fraud") == 1
    )
)

normal_transactions = sum(
    1
    for node, attributes
    in G.nodes(data=True)
    if (
        attributes.get("node_type")
        == "transaction"
        and attributes.get("is_fraud") == 0
    )
)

print("\nTransaction classification:")

print(
    f"  Normal transactions: "
    f"{normal_transactions:,}"
)

print(
    f"  Fraud transactions:  "
    f"{fraud_transactions:,}"
)


print("\nGraph saved to:")

print(graph_file)

print("\nEdge list saved to:")

print(edge_file)

print("\nNext step:")
print(
    "Visualize suspicious fraud relationships "
    "using NetworkX."
)

print("=" * 60)