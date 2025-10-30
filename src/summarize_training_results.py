import os
import re
import pandas as pd

# === Directories ===
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# === Regex patterns ===
epoch_pattern = re.compile(
    r"Epoch\s+(\d+):\s+train loss ([\deE\+\-\.]+), val loss ([\deE\+\-\.]+), train acc\s+([\d\.]+), val acc\s+([\d\.]+).*?\(([\d\.]+)s\)"
)
dataset_pattern = re.compile(r"Using datasets\s+\['([^']+)'\]")
loss_pattern = re.compile(r"Using loss function ([A-Za-z0-9_]+)")
epoch_count_pattern = re.compile(r"Using (\d+) epochs")

# === Storage ===
records = []

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

    # Find all epoch metrics
    matches = epoch_pattern.findall(text)
    if not matches:
        print(f"⚠️ No epoch data found in {fname}, skipping.")
        continue

    # Use last epoch (or "best" if we later add early stopping)
    last_epoch = matches[-1]
    epoch_num = int(last_epoch[0])
    train_loss = float(last_epoch[1])
    val_loss = float(last_epoch[2])
    train_acc = float(last_epoch[3])
    val_acc = float(last_epoch[4])
    epoch_time = float(last_epoch[5])

    records.append({
        "Dataset": dataset,
        "Loss Function": loss_name,
        "Epochs": total_epochs,
        "Last Epoch": epoch_num,
        "Train Loss": train_loss,
        "Val Loss": val_loss,
        "Train Acc": train_acc,
        "Val Acc": val_acc,
        "Epoch Time (s)": epoch_time,
        "File": fname
    })

# === Make a table ===
if records:
    df = pd.DataFrame(records)
    df = df.sort_values(["Dataset", "Loss Function"]).reset_index(drop=True)
    print("\n📊 Training Summary:\n")
    print(df.to_string(index=False, justify="center", float_format=lambda x: f"{x:.3f}"))
else:
    print("⚠️ No training summaries found in this directory.")
