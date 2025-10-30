import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Directory setup ===
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(SRC_DIR, "../output_dir/plots"))
os.makedirs(OUT_DIR, exist_ok=True)

# === Regex patterns ===
epoch_pattern = re.compile(
    r"Epoch\s+(\d+):\s+train loss ([\deE\+\-\.]+), val loss ([\deE\+\-\.]+), train acc\s+([\d\.]+), val acc\s+([\d\.]+)"
)
dataset_pattern = re.compile(r"Using datasets\s+\['([^']+)'\]")
loss_pattern = re.compile(r"Using loss function ([A-Za-z0-9_]+)")
epoch_count_pattern = re.compile(r"Using (\d+) epochs")

# === Loop through all .out files in src ===
for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".out"):
        continue

    fpath = os.path.join(SRC_DIR, fname)
    with open(fpath, "r") as f:
        text = f.read()

    # Extract metadata
    dataset_match = dataset_pattern.search(text)
    loss_match = loss_pattern.search(text)
    epoch_count_match = epoch_count_pattern.search(text)

    dataset = dataset_match.group(1) if dataset_match else "UnknownDataset"
    loss_name = loss_match.group(1) if loss_match else "UnknownLoss"
    total_epochs = int(epoch_count_match.group(1)) if epoch_count_match else 0

    # Extract epoch metrics
    matches = epoch_pattern.findall(text)
    if not matches:
        print(f"⚠️ No epoch data found in {fname}, skipping.")
        continue

    epochs = [int(m[0]) for m in matches]
    train_loss = [float(m[1]) for m in matches]
    val_loss = [float(m[2]) for m in matches]
    train_acc = [float(m[3]) for m in matches]
    val_acc = [float(m[4]) for m in matches]

    # === PLOTTING ===
    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset} – {total_epochs} epochs – Loss Curve ({loss_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_loss_path = os.path.join(OUT_DIR, f"{dataset}_{loss_name}_loss.png")
    plt.savefig(out_loss_path, dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{dataset} – {total_epochs} epochs – Accuracy Curve ({loss_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_acc_path = os.path.join(OUT_DIR, f"{dataset}_{loss_name}_acc.png")
    plt.savefig(out_acc_path, dpi=200)
    plt.close()

    print(f"✅ Plots saved for {fname}:")
    print(f"   - {out_loss_path}")
    print(f"   - {out_acc_path}\n")

print("🎉 Done! All plots saved in:", OUT_DIR)
