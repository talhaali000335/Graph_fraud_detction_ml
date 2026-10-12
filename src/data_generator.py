"""
Graph Fraud Detection AI
Synthetic Telecom/E-Commerce Fraud Graph Dataset Generator

Creates:
- Customers
- SIM cards
- Devices
- IP addresses
- Merchants
- Transactions
- Normal relationships
- Suspicious shared-device/shared-IP behaviour
- Fraud rings

Output:
    data/raw/
        customers.csv
        sims.csv
        devices.csv
        ips.csv
        merchants.csv
        transactions.csv
        fraud_rings.csv
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

SEED = 42

N_CUSTOMERS = 10_000
N_SIMS = 10_000
N_DEVICES = 8_000
N_IPS = 7_000
N_MERCHANTS = 500
N_TRANSACTIONS = 30_000

N_FRAUD_RINGS = 50
FRAUD_RING_SIZE_MIN = 4
FRAUD_RING_SIZE_MAX = 8


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def random_ip():
    """Generate a private-style IPv4 address."""

    return (
        f"10.{random.randint(0, 255)}."
        f"{random.randint(0, 255)}."
        f"{random.randint(1, 254)}"
    )


def random_phone():
    """Generate a synthetic MSISDN."""

    return f"973{random.randint(30000000, 39999999)}"


def random_imei():
    """Generate a synthetic IMEI-like identifier."""

    return str(random.randint(100000000000000, 999999999999999))


# ============================================================
# 1. Customers
# ============================================================

print("\n[1/8] Generating customers...")

nationalities = [
    "Bahraini",
    "Indian",
    "Pakistani",
    "Bangladeshi",
    "Filipino",
    "Egyptian",
    "Jordanian",
    "Other",
]

occupations = [
    "Engineer",
    "Manager",
    "Student",
    "Teacher",
    "Driver",
    "Sales",
    "IT",
    "Healthcare",
    "Government",
    "Business Owner",
    "Other",
]

customers = pd.DataFrame({
    "customer_id": [
        f"CUST_{i:05d}"
        for i in range(1, N_CUSTOMERS + 1)
    ],
    "age": np.random.randint(18, 70, N_CUSTOMERS),
    "nationality": np.random.choice(
        nationalities,
        N_CUSTOMERS,
        p=[
            0.15,
            0.25,
            0.20,
            0.10,
            0.08,
            0.08,
            0.05,
            0.09,
        ],
    ),
    "occupation": np.random.choice(
        occupations,
        N_CUSTOMERS,
    ),
    "account_age_days": np.random.randint(
        30,
        3650,
        N_CUSTOMERS,
    ),
})


# ============================================================
# 2. SIM cards
# ============================================================

print("[2/8] Generating SIM cards...")

sims = pd.DataFrame({
    "sim_id": [
        f"SIM_{i:05d}"
        for i in range(1, N_SIMS + 1)
    ],
    "customer_id": customers["customer_id"],
    "msisdn": [
        random_phone()
        for _ in range(N_SIMS)
    ],
})


# ============================================================
# 3. Devices
# ============================================================

print("[3/8] Generating devices...")

device_brands = [
    "Apple",
    "Samsung",
    "Huawei",
    "Xiaomi",
    "Google",
    "Other",
]

devices = pd.DataFrame({
    "device_id": [
        f"DEV_{i:05d}"
        for i in range(1, N_DEVICES + 1)
    ],
    "brand": np.random.choice(
        device_brands,
        N_DEVICES,
        p=[
            0.30,
            0.30,
            0.12,
            0.12,
            0.06,
            0.10,
        ],
    ),
    "imei": [
        random_imei()
        for _ in range(N_DEVICES)
    ],
})


# ============================================================
# 4. IP addresses
# ============================================================

print("[4/8] Generating IP addresses...")

ips = pd.DataFrame({
    "ip_id": [
        f"IP_{i:05d}"
        for i in range(1, N_IPS + 1)
    ],
    "ip_address": [
        random_ip()
        for _ in range(N_IPS)
    ],
    "country": np.random.choice(
        [
            "Bahrain",
            "UAE",
            "Saudi Arabia",
            "India",
            "Pakistan",
            "Other",
        ],
        N_IPS,
        p=[
            0.50,
            0.12,
            0.12,
            0.10,
            0.08,
            0.08,
        ],
    ),
})


# ============================================================
# 5. Merchants
# ============================================================

print("[5/8] Generating merchants...")

merchant_categories = [
    "Electronics",
    "Mobile Phones",
    "Clothing",
    "Groceries",
    "Travel",
    "Gaming",
    "Digital Services",
    "Other",
]

merchants = pd.DataFrame({
    "merchant_id": [
        f"MER_{i:04d}"
        for i in range(1, N_MERCHANTS + 1)
    ],
    "category": np.random.choice(
        merchant_categories,
        N_MERCHANTS,
    ),
})


# ============================================================
# 6. Normal customer relationships
# ============================================================

print("[6/8] Creating normal relationships...")

customer_ids = customers["customer_id"].tolist()
sim_ids = sims["sim_id"].tolist()
device_ids = devices["device_id"].tolist()
ip_ids = ips["ip_id"].tolist()
merchant_ids = merchants["merchant_id"].tolist()


# Each customer normally has one SIM
customer_to_sim = {
    customer_ids[i]: sim_ids[i]
    for i in range(N_CUSTOMERS)
}


# Assign devices to customers.
#
# Because there are fewer devices than customers,
# some devices will naturally be shared.
customer_to_device = {}

for customer_id in customer_ids:

    # Mostly use one device.
    device_id = random.choice(device_ids)

    customer_to_device[customer_id] = device_id


# Assign normal IPs.
customer_to_ip = {}

for customer_id in customer_ids:

    ip_id = random.choice(ip_ids)

    customer_to_ip[customer_id] = ip_id


# ============================================================
# 7. Generate transactions
# ============================================================

print("[7/8] Generating transactions...")

transactions = []

for transaction_number in range(1, N_TRANSACTIONS + 1):

    customer_id = random.choice(customer_ids)

    sim_id = customer_to_sim[customer_id]
    device_id = customer_to_device[customer_id]
    ip_id = customer_to_ip[customer_id]

    merchant_id = random.choice(merchant_ids)

    # Transaction amount.
    amount = round(
        np.random.lognormal(
            mean=4.0,
            sigma=0.8,
        ),
        2,
    )

    transaction = {
        "transaction_id": f"TXN_{transaction_number:06d}",
        "customer_id": customer_id,
        "sim_id": sim_id,
        "device_id": device_id,
        "ip_id": ip_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "is_fraud": 0,
        "fraud_type": "Normal",
    }

    transactions.append(transaction)


transactions = pd.DataFrame(transactions)


# ============================================================
# 8. Inject fraud rings
# ============================================================

print("[8/8] Injecting fraud rings...")

fraud_ring_records = []

available_customers = customer_ids.copy()

random.shuffle(available_customers)


for ring_number in range(1, N_FRAUD_RINGS + 1):

    ring_size = random.randint(
        FRAUD_RING_SIZE_MIN,
        FRAUD_RING_SIZE_MAX,
    )

    if len(available_customers) < ring_size:
        break

    ring_customers = available_customers[:ring_size]

    available_customers = available_customers[ring_size:]

    # Fraud ring shares one device and one IP.
    shared_device = random.choice(device_ids)
    shared_ip = random.choice(ip_ids)

    for customer_id in ring_customers:

        # Update relationships.
        customer_to_device[customer_id] = shared_device
        customer_to_ip[customer_id] = shared_ip

        fraud_ring_records.append({
            "fraud_ring_id": f"RING_{ring_number:03d}",
            "customer_id": customer_id,
            "shared_device_id": shared_device,
            "shared_ip_id": shared_ip,
        })

    # Generate fraudulent transactions for the ring.
    for _ in range(
        random.randint(10, 25)
    ):

        customer_id = random.choice(ring_customers)

        merchant_id = random.choice(merchant_ids)

        amount = round(
            np.random.uniform(
                300,
                2500,
            ),
            2,
        )

        transaction = {
            "transaction_id": (
                f"FRAUD_TXN_{ring_number:03d}_"
                f"{random.randint(10000, 99999)}"
            ),
            "customer_id": customer_id,
            "sim_id": customer_to_sim[customer_id],
            "device_id": shared_device,
            "ip_id": shared_ip,
            "merchant_id": merchant_id,
            "amount": amount,
            "is_fraud": 1,
            "fraud_type": random.choice([
                "Fraud Ring",
                "Shared Device",
                "Shared IP",
            ]),
        }

        transactions = pd.concat(
            [
                transactions,
                pd.DataFrame([transaction]),
            ],
            ignore_index=True,
        )


fraud_rings = pd.DataFrame(fraud_ring_records)


# ============================================================
# Update customer-device and customer-IP relationships
# ============================================================

customer_relationships = pd.DataFrame({
    "customer_id": customer_ids,
    "sim_id": [
        customer_to_sim[c]
        for c in customer_ids
    ],
    "device_id": [
        customer_to_device[c]
        for c in customer_ids
    ],
    "ip_id": [
        customer_to_ip[c]
        for c in customer_ids
    ],
})


# ============================================================
# Save datasets
# ============================================================

print("\nSaving datasets...")

customers.to_csv(
    RAW_DATA_DIR / "customers.csv",
    index=False,
)

sims.to_csv(
    RAW_DATA_DIR / "sims.csv",
    index=False,
)

devices.to_csv(
    RAW_DATA_DIR / "devices.csv",
    index=False,
)

ips.to_csv(
    RAW_DATA_DIR / "ips.csv",
    index=False,
)

merchants.to_csv(
    RAW_DATA_DIR / "merchants.csv",
    index=False,
)

transactions.to_csv(
    RAW_DATA_DIR / "transactions.csv",
    index=False,
)

fraud_rings.to_csv(
    RAW_DATA_DIR / "fraud_rings.csv",
    index=False,
)

customer_relationships.to_csv(
    RAW_DATA_DIR / "customer_relationships.csv",
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("GRAPH FRAUD DATASET GENERATION COMPLETE")
print("=" * 60)

print(f"\nCustomers:        {len(customers):,}")
print(f"SIMs:             {len(sims):,}")
print(f"Devices:          {len(devices):,}")
print(f"IP addresses:     {len(ips):,}")
print(f"Merchants:        {len(merchants):,}")
print(f"Transactions:     {len(transactions):,}")

print(
    f"\nFraud transactions: "
    f"{transactions['is_fraud'].sum():,}"
)

fraud_rate = (
    transactions["is_fraud"].mean() * 100
)

print(
    f"Fraud rate:        "
    f"{fraud_rate:.2f}%"
)

print(
    f"\nFraud rings:       "
    f"{fraud_rings['fraud_ring_id'].nunique():,}"
)

print(
    f"Fraud-ring members:"
    f" {len(fraud_rings):,}"
)

print("\nOutput directory:")
print(RAW_DATA_DIR)

print("\nFiles created:")

for file in sorted(RAW_DATA_DIR.glob("*.csv")):
    print(f"  ✓ {file.name}")

print("\nNext step:")
print("Build the graph using NetworkX + PyTorch Geometric.")
print("=" * 60)