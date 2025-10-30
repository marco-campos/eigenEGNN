import os
import re
import argparse
import matplotlib.pyplot as plt

# === Argument parser ===
parser = argparse.ArgumentParser(description="Plot loss or accuracy curves for multiple datasets and losses.")
parser.add_argument("--data", nargs="+", required=True, help="Datasets to include (e.g. ModelNet10 Tori)")
parser.add_argument("--loss", nargs="+", required=True, help="Loss functions to include (e.g. L1 L2 Polar RPD)")
parser.add_argument("--metric", choices=["loss", "acc"], default="loss",
                    help="Metric to plot: 'loss' or 'acc' (default: loss)")
args = parser.parse_args()

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

# === Parse all .out files ===
results = {}  # results[dataset][loss_name] = (epochs, train_loss, val_loss, train_acc, val_acc)
for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".out"):
        continue

    fpath = os.path.join(SRC_DIR, fname)
    with open(fpath, "r") as f:
        text = f.read()

    dataset_match = dataset_pattern.search(text)
    loss_match = loss_pattern.search(text)
    if not dataset_match or not loss_match:
        continue

    dataset = dataset_match.group(1)
    loss_name = loss_match.group(1)
    if dataset not in args.data or loss_name not in args.loss:
        continue

    matches = epoch_pattern.findall(text)
    if not matches:
        print(f"⚠️ No epoch data found in {fname}, skipping.")
        continue

    epochs = [int(m[0]) for m in matches]
    train_loss = [float(m[1]) for m in matches]
    val_loss = [float(m[2]) for m in matches]
    train_acc = [float(m[3]) for m in matches]
    val_acc = [float(m[4]) for m in matches]

    results.setdefault(dataset, {})[loss_name] = (epochs, train_loss, val_loss, train_acc, val_acc)

# === Plot grid ===
num_datasets = len(args.data)
num_losses = len(args.loss)
fig, axes = plt.subplots(num_datasets, num_losses, figsize=(4*num_losses, 3*num_datasets), squeeze=False)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, dataset in enumerate(args.data):
    for j, loss_name in enumerate(args.loss):
        ax = axes[i, j]
        if dataset in results and loss_name in results[dataset]:
            epochs, train_loss, val_loss, train_acc, val_acc = results[dataset][loss_name]
            if args.metric == "loss":
                ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
                ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)
                ax.set_ylabel("Loss")
            else:  # accuracy
                ax.plot(epochs, train_acc, label="Train Acc", linewidth=2)
                ax.plot(epochs, val_acc, label="Val Acc", linewidth=2)
                ax.set_ylabel("Accuracy (%)")

            ax.set_title(f"{dataset} – {loss_name}", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])

# === Save ===
metric_name = "accuracy" if args.metric == "acc" else "loss"
out_path = os.path.join(OUT_DIR, f"combined_{metric_name}_grid.png")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()

print(f"✅ Combined {metric_name} grid saved at: {out_path}")
