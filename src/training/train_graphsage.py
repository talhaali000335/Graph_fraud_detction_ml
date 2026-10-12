"""
Graph Fraud Detection AI
Step 8: Train GraphSAGE

Trains a two-layer GraphSAGE model for transaction
fraud classification.

Target:
    0 = Normal
    1 = Fraud

Evaluation:
    Accuracy
    Precision
    Recall
    F1
    ROC-AUC
    Average Precision
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

import torch.nn.functional as F
from torch.optim import Adam


# ============================================================
# Make src importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


from models.graphsage import (
    GraphSAGEFraudDetector,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

HIDDEN_DIM = 64

LEARNING_RATE = 0.005

WEIGHT_DECAY = 1e-4

EPOCHS = 100

PATIENCE = 15

DROPOUT = 0.30


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Paths
# ============================================================

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATA_FILE = (
    DATA_DIR /
    "fraud_graph_pyg.pt"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n" + "=" * 60)
print("GRAPHSAGE FRAUD DETECTION")
print("=" * 60)

print(
    f"\nTraining device: {device}"
)

if device.type == "cpu":

    print(
        "CUDA is not available."
    )

    print(
        "Training will use the CPU."
    )


# ============================================================
# Load PyG graph
# ============================================================

print("\nLoading graph dataset...")

data = torch.load(
    DATA_FILE,
    weights_only=False,
)

data = data.to(device)


print(
    f"Nodes: {data.num_nodes:,}"
)

print(
    f"Edges: {data.num_edges:,}"
)

print(
    f"Features: {data.num_node_features}"
)


# ============================================================
# Model
# ============================================================

model = GraphSAGEFraudDetector(
    input_dim=data.num_node_features,
    hidden_dim=HIDDEN_DIM,
    output_dim=2,
    dropout=DROPOUT,
).to(device)


print("\nModel architecture:")
print(model)


# ============================================================
# Class weights
# ============================================================

train_labels = data.y[
    data.train_mask
]

normal_count = (
    train_labels == 0
).sum().item()

fraud_count = (
    train_labels == 1
).sum().item()


# Balanced weighting:
#
# weight for class 0 = total / (2 * normal)
# weight for class 1 = total / (2 * fraud)

total_train = (
    normal_count +
    fraud_count
)

normal_weight = (
    total_train /
    (2 * normal_count)
)

fraud_weight = (
    total_train /
    (2 * fraud_count)
)


class_weights = torch.tensor(
    [
        normal_weight,
        fraud_weight,
    ],
    dtype=torch.float,
    device=device,
)


print("\nClass weights:")

print(
    f"  Normal: {normal_weight:.4f}"
)

print(
    f"  Fraud:  {fraud_weight:.4f}"
)


# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Training
# ============================================================

best_val_f1 = -1.0

best_state = None

patience_counter = 0

training_history = []


print("\nStarting training...")

print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1,
):

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    optimizer.zero_grad()

    logits = model(
        data.x,
        data.edge_index,
    )

    train_logits = logits[
        data.train_mask
    ]

    train_targets = data.y[
        data.train_mask
    ]

    loss = F.cross_entropy(
        train_logits,
        train_targets,
        weight=class_weights,
    )

    loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        all_logits = model(
            data.x,
            data.edge_index,
        )

        val_logits = all_logits[
            data.val_mask
        ]

        val_targets = data.y[
            data.val_mask
        ]

        val_probabilities = torch.softmax(
            val_logits,
            dim=1,
        )[:, 1]

        val_predictions = (
            val_probabilities >= 0.50
        ).long()


    val_true = (
        val_targets
        .detach()
        .cpu()
        .numpy()
    )

    val_pred = (
        val_predictions
        .detach()
        .cpu()
        .numpy()
    )

    val_prob = (
        val_probabilities
        .detach()
        .cpu()
        .numpy()
    )


    val_f1 = f1_score(
        val_true,
        val_pred,
        zero_division=0,
    )

    val_recall = recall_score(
        val_true,
        val_pred,
        zero_division=0,
    )

    val_precision = precision_score(
        val_true,
        val_pred,
        zero_division=0,
    )

    val_auc = roc_auc_score(
        val_true,
        val_prob,
    )


    training_history.append({
        "epoch": epoch,
        "loss": loss.item(),
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_roc_auc": val_auc,
    })


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_state = {
            key: value.detach().cpu().clone()
            for key, value
            in model.state_dict().items()
        }

        patience_counter = 0

    else:

        patience_counter += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 5 == 0
    ):

        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.4f} | "
            f"Precision {val_precision:.4f} | "
            f"Recall {val_recall:.4f} | "
            f"F1 {val_f1:.4f} | "
            f"ROC-AUC {val_auc:.4f}"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            f"\nEarly stopping at epoch "
            f"{epoch}."
        )

        break


# ============================================================
# Restore best model
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# Final test evaluation
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


model.eval()

with torch.no_grad():

    logits = model(
        data.x,
        data.edge_index,
    )

    test_logits = logits[
        data.test_mask
    ]

    test_targets = data.y[
        data.test_mask
    ]

    test_probabilities = torch.softmax(
        test_logits,
        dim=1,
    )[:, 1]

    test_predictions = (
        test_probabilities >= 0.50
    ).long()


# Move to CPU

y_true = (
    test_targets
    .detach()
    .cpu()
    .numpy()
)

y_pred = (
    test_predictions
    .detach()
    .cpu()
    .numpy()
)

y_prob = (
    test_probabilities
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_true,
    y_prob,
)

average_precision = (
    average_precision_score(
        y_true,
        y_prob,
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)


# ============================================================
# Print results
# ============================================================

print(
    f"\nAccuracy:          {accuracy:.4f}"
)

print(
    f"Precision:         {precision:.4f}"
)

print(
    f"Recall:            {recall:.4f}"
)

print(
    f"F1 Score:          {f1:.4f}"
)

print(
    f"ROC-AUC:           {roc_auc:.4f}"
)

print(
    f"Average Precision: {average_precision:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


print("\nTest distribution:")

print(
    f"  Normal: "
    f"{(y_true == 0).sum():,}"
)

print(
    f"  Fraud:  "
    f"{(y_true == 1).sum():,}"
)


# ============================================================
# Save metrics
# ============================================================

metrics = pd.DataFrame([
    {
        "model": "GraphSAGE",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "best_validation_f1": best_val_f1,
    }
])


metrics_file = (
    RESULTS_DIR /
    "graphsage_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False,
)


# ============================================================
# Save predictions
# ============================================================

predictions = pd.DataFrame({
    "true_label": y_true,
    "predicted_label": y_pred,
    "fraud_probability": y_prob,
})


predictions_file = (
    RESULTS_DIR /
    "graphsage_predictions.csv"
)

predictions.to_csv(
    predictions_file,
    index=False,
)


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(
    training_history
)

history_file = (
    RESULTS_DIR /
    "graphsage_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False,
)


# ============================================================
# Save model
# ============================================================

model_file = (
    RESULTS_DIR /
    "graphsage_fraud_model.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_dim":
            data.num_node_features,

        "hidden_dim":
            HIDDEN_DIM,

        "output_dim":
            2,

        "dropout":
            DROPOUT,

        "best_validation_f1":
            best_val_f1,
    },
    model_file,
)


# ============================================================
# Complete
# ============================================================

print("\nSaved results:")

print(
    f"  ✓ {metrics_file}"
)

print(
    f"  ✓ {predictions_file}"
)

print(
    f"  ✓ {history_file}"
)

print(
    f"  ✓ {model_file}"
)


print("\n" + "=" * 60)
print("GRAPHSAGE TRAINING COMPLETE")
print("=" * 60)

print(
    "\nNext step:"
)

print(
    "Compare GraphSAGE against a traditional "
    "non-graph fraud model."
)