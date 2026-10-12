"""
Graph Fraud Detection AI
Fraud Ring Visualization

Finds a known synthetic fraud ring and creates
a visual representation of its graph relationships.
"""

from pathlib import Path

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load data
# ============================================================

print("\nLoading fraud ring data...")

fraud_rings = pd.read_csv(
    RAW_DATA_DIR / "fraud_rings.csv"
)

transactions = pd.read_csv(
    RAW_DATA_DIR / "transactions.csv"
)


# ============================================================
# Select one fraud ring
# ============================================================

ring_id = fraud_rings[
    "fraud_ring_id"
].iloc[0]

print(
    f"Selected fraud ring: {ring_id}"
)


ring = fraud_rings[
    fraud_rings["fraud_ring_id"] == ring_id
].copy()


print(
    f"Customers in ring: {len(ring)}"
)


# ============================================================
# Build subgraph
# ============================================================

G = nx.Graph()


# ------------------------------------------------------------
# Add customers
# ------------------------------------------------------------

for customer_id in ring["customer_id"]:

    G.add_node(
        customer_id,
        node_type="customer",
    )


# ------------------------------------------------------------
# Add shared device
# ------------------------------------------------------------

shared_device = ring[
    "shared_device_id"
].iloc[0]

G.add_node(
    shared_device,
    node_type="device",
)


# ------------------------------------------------------------
# Add shared IP
# ------------------------------------------------------------

shared_ip = ring[
    "shared_ip_id"
].iloc[0]

G.add_node(
    shared_ip,
    node_type="ip",
)


# ------------------------------------------------------------
# Connect customers to shared device and IP
# ------------------------------------------------------------

for customer_id in ring["customer_id"]:

    G.add_edge(
        customer_id,
        shared_device,
        relationship="shared_device",
    )

    G.add_edge(
        customer_id,
        shared_ip,
        relationship="shared_ip",
    )


# ============================================================
# Add fraudulent transactions
# ============================================================

ring_customers = set(
    ring["customer_id"]
)

ring_transactions = transactions[
    (
        transactions["customer_id"]
        .isin(ring_customers)
    )
    &
    (
        transactions["is_fraud"] == 1
    )
].copy()


print(
    f"Fraudulent transactions in ring: "
    f"{len(ring_transactions)}"
)


for _, transaction in ring_transactions.iterrows():

    transaction_id = transaction[
        "transaction_id"
    ]

    merchant_id = transaction[
        "merchant_id"
    ]

    G.add_node(
        transaction_id,
        node_type="transaction",
    )

    G.add_node(
        merchant_id,
        node_type="merchant",
    )

    G.add_edge(
        transaction["customer_id"],
        transaction_id,
        relationship="fraud_transaction",
    )

    G.add_edge(
        transaction_id,
        merchant_id,
        relationship="merchant",
    )


# ============================================================
# Print graph statistics
# ============================================================

print("\n" + "=" * 60)
print("FRAUD RING SUBGRAPH")
print("=" * 60)

print(
    f"Nodes: {G.number_of_nodes()}"
)

print(
    f"Edges: {G.number_of_edges()}"
)

print(
    f"Customers: "
    f"{sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'customer')}"
)

print(
    f"Transactions: "
    f"{sum(1 for _, d in G.nodes(data=True) if d.get('node_type') == 'transaction')}"
)

print(
    f"Shared device: {shared_device}"
)

print(
    f"Shared IP: {shared_ip}"
)


# ============================================================
# Visualize
# ============================================================

print("\nCreating visualization...")


plt.figure(
    figsize=(16, 12)
)


# Spring layout
pos = nx.spring_layout(
    G,
    seed=42,
    k=1.5,
)


# Identify node types

customers_nodes = [
    node
    for node, data
    in G.nodes(data=True)
    if data.get("node_type") == "customer"
]

device_nodes = [
    node
    for node, data
    in G.nodes(data=True)
    if data.get("node_type") == "device"
]

ip_nodes = [
    node
    for node, data
    in G.nodes(data=True)
    if data.get("node_type") == "ip"
]

transaction_nodes = [
    node
    for node, data
    in G.nodes(data=True)
    if data.get("node_type") == "transaction"
]

merchant_nodes = [
    node
    for node, data
    in G.nodes(data=True)
    if data.get("node_type") == "merchant"
]


# Draw edges

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.35,
    width=1.2,
)


# Draw customers

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=customers_nodes,
    node_size=500,
    node_color="skyblue",
    node_shape="o",
    label="Customers",
)


# Draw device

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=device_nodes,
    node_size=1000,
    node_color="orange",
    node_shape="s",
    label="Shared Device",
)


# Draw IP

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=ip_nodes,
    node_size=1000,
    node_color="purple",
    node_shape="D",
    label="Shared IP",
)


# Draw transactions

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=transaction_nodes,
    node_size=250,
    node_color="red",
    node_shape="^",
    label="Fraud Transactions",
)


# Draw merchants

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=merchant_nodes,
    node_size=300,
    node_color="green",
    node_shape="P",
    label="Merchants",
)


# Labels

nx.draw_networkx_labels(
    G,
    pos,
    font_size=7,
)


plt.title(
    f"Graph Fraud Detection AI — Fraud Ring {ring_id}",
    fontsize=16,
)


plt.legend(
    scatterpoints=1
)


plt.axis("off")


# ============================================================
# Save
# ============================================================

output_file = (
    RESULTS_DIR /
    f"{ring_id}_visualization.png"
)

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.show()


print(
    f"\nVisualization saved to:"
)

print(output_file)

print("\nStep 6 complete.")