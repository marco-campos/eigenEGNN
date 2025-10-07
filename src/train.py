# train.py
import argparse
import datetime
import json
import os
import os.path as osp
import random
import warnings
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# --- your utilities & settings ---
from accuracy import (
    PSNR, RPD, PolarLoss, Polar2Loss, RPDLoss, SmoothRPDLoss,
    inv_lap, l2_error, polar_loss,
)
from egnn import E_GCL, EGNN
from settings import NUM_EIGS, DEVICE

mp.set_start_method('spawn', force=True)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def safe_load_pt(pt_path):
    """Handle PyTorch 2.6 'weights_only=True' default when loading a list[Data]."""
    try:
        from torch_geometric.data import Data
        try:
            from torch_geometric.data.data import DataEdgeAttr
            safe_list = [Data, DataEdgeAttr]
        except Exception:
            safe_list = [Data]
        with torch.serialization.safe_globals(safe_list):
            return torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception:
        # If that fails (old torch), just force weights_only=False.
        return torch.load(pt_path, map_location="cpu", weights_only=False)


def build_loaders_from_pt(pt_path, batch_size, seed=0xC0FFEE):
    graphs = safe_load_pt(pt_path)
    if not isinstance(graphs, (list, tuple)) or len(graphs) == 0:
        raise ValueError(f"Loaded object from {pt_path} is empty or not a list[Data].")

    total = len(graphs)
    train_size = int(total * 0.8)
    val_size   = int(total * 0.1)
    test_size  = total - train_size - val_size
    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(graphs, [train_size, val_size, test_size], generator=g)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1,         shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1,         shuffle=False, num_workers=4, pin_memory=True)

    return graphs, train_loader, val_loader, test_loader


def init_egnn(in_channels, out_channels, hidden_channels, mlp_hidden, mlp_layers, device):
    model = EGNN(
        input_channels=in_channels,
        output_channels=out_channels,
        hidden_channels=hidden_channels,
        MLP_hidden_channels=mlp_hidden,
        MLP_num_layers=mlp_layers,
        edge_features_dim=0,
        coords_agg='mean',
        act_fn=torch.nn.SiLU()
    ).to(device)
    return model


def pick_optimizer_and_scheduler(model, lr, scheduler_choice):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-5,
    )
    if scheduler_choice == "Cos":
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-8)
    elif scheduler_choice == "Exp":
        scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.988)
    else:
        raise ValueError("Scheduler must be 'Cos' or 'Exp'")
    return optimizer, scheduler1


def pick_criterion(loss_name):
    loss_name = loss_name.upper()
    if loss_name == "RPD":
        return RPDLoss()
    elif loss_name == "POLAR2":
        return Polar2Loss(lmbda=1, reduction="mean")
    elif loss_name == "POLAR":
        return PolarLoss(lmbda=1, reduction="mean")
    elif loss_name == "L1":
        return L1Loss()
    elif loss_name == "L2":
        return MSELoss()
    else:
        raise ValueError("loss_fn must be one of: RPD, Polar, Polar2, L1, L2")


def run(
    workspace="/scratch/bdbq/mcampos1/eigenEGNN_model",
    output_dir="models",
    batch_size=32,
    datasets=("Tori",),
    scheduler="Cos",
    loss_fn="L2",
    lr=1e-3,
    epochs=5,
    model_type="EGNN",
    hidden_channels=(64, 128, 256),
    MLP_hidden_channels=64,
    MLP_num_layers=5,
):
    print("The job started at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    print(f"Using datasets {list(datasets)}")
    print(f"Using scheduler {scheduler}...")
    print(f"Using loss function {loss_fn}...")
    print(f"Using initial learning rate {lr}...")
    print(f"Using batch size {batch_size}...")
    print(f"Using {epochs} epochs...")
    print(f"Using model {model_type} with hidden layers {list(hidden_channels)} and "
          f"MLP(channels={MLP_hidden_channels}, layers={MLP_num_layers})...")

    # seeds
    np.random.seed(0xDEADBEEF)
    random.seed(0xDEADBEEF)
    torch.manual_seed(0xDEADBEEF)
    torch.cuda.manual_seed(0xDEADBEEF)

    # pick the cached dataset
    dataset_name = datasets[0]
    preferred_path = osp.join(workspace, "data_processed", f"processed_{dataset_name}.pt")
    if osp.exists(preferred_path):
        pt_path = preferred_path
    else:
        cands = list(Path(workspace).rglob("*.pt"))
        if not cands:
            raise FileNotFoundError(f"No .pt dataset found at {preferred_path} or anywhere under {workspace}.")
        pt_path = str(cands[0])
    print(f"[info] Using cached dataset: {pt_path}")

    # load graphs + loaders
    graphs, train_loader, val_loader, test_loader = build_loaders_from_pt(pt_path, batch_size=batch_size)

    g0 = graphs[0]
    print(f"[info] First graph: nodes={g0.num_nodes}, edges={g0.num_edges}, "
          f"x={tuple(g0.x.shape) if g0.x is not None else None}, pos={tuple(g0.pos.shape)}")

    t2 = time.time()
    print(f"Datasets loaded in {t2 - start_time:.2f} seconds.")

    MODEL_SAVE_DIR = f"{workspace}/{output_dir}"
    print(f"Saving all models to {MODEL_SAVE_DIR}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    datasets_tag = "-".join(datasets)
    run_dir = os.path.join(MODEL_SAVE_DIR, f"{datasets_tag}_{model_type}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump({
            "datasets": list(datasets),
            "model_type": model_type,
            "hidden_channels": list(hidden_channels),
            "mlp_hidden": MLP_hidden_channels,
            "mlp_layers": MLP_num_layers,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "scheduler": scheduler,
            "loss_fn": loss_fn,
            "device": str(DEVICE),
            "dataset_pt": pt_path,
        }, f, indent=2)
    print(f"Run dir: {run_dir}")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # infer in/out channels from your task
    in_channels = g0.x.size(1) if g0.x is not None else 1
    out_channels = NUM_EIGS  # <- set to your task (e.g., NUM_EIGS). Change as needed.

    # init model
    model = init_egnn(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=list(hidden_channels),
        mlp_hidden=MLP_hidden_channels,
        mlp_layers=MLP_num_layers,
        device=device,
    )
    print("Model loaded")

    # optimizer & scheduler & criterion
    optimizer, scheduler1 = pick_optimizer_and_scheduler(model, lr, scheduler)
    criterion = pick_criterion(loss_fn)
    print("Optimizer and scheduler loaded")

    QUANTILES = torch.tensor([0, 0.25, 0.5, 0.75, 1], device=device)
    print("Auxiliary loaded")
    if torch.cuda.is_available():
        print(f"torch version: {torch.__version__}. using device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"torch version: {torch.__version__}. using CPU")

    # ---------- train & test ----------
    def forward_for_task(data):
        # EGNN path: edge_index + pos
        return model(data.x, data.edge_index, data.batch, pos=data.pos)

    def train_epoch(loader):
        model.train()
        size_count = 0
        loss_accum = 0.0
        rpd_sum = 0.0
        polar_sum = 0.0
        psnr_list = []
        inv_lap_sum = 0.0
        l1_sum = 0.0
        l2_sum = 0.0

        for data in loader:
            data = data.to(device, non_blocking=True)
            out = forward_for_task(data)
            n = out.size(0)

            # Expect data.y as graph-level labels; if missing, skip
            if getattr(data, "y", None) is None:
                print("[warn] Batch has no 'y' labels; skipping (flow-only).")
                continue

            target = torch.reshape(data.y, (n, -1)).to(device)
            loss = criterion(out, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # metrics (optional; safe even for L1/L2)
            with torch.no_grad():
                loss_accum += loss.item() * n
                size_count += n
                try:
                    ps = PSNR(out, target)
                    psnr_list.append(ps)
                    rpd_sum += RPD(out, target).sum().item()
                    polar_sum += polar_loss(out, target, lmbda=1).sum().item()
                    inv_lap_sum += inv_lap(out, target).sum().item()
                    l1_sum += F.l1_loss(out, target, reduction="sum").item()
                    l2_sum += l2_error(out, target).sum().item()
                except Exception:
                    pass

            torch.cuda.empty_cache()

        if size_count == 0:
            return (float('nan'), 0.0, float('nan'), float('nan'), "[n/a]",
                    float('nan'), float('nan'), float('nan'), float('nan'))

        psnrs_t = torch.cat(psnr_list) if len(psnr_list) else torch.tensor([], device=device)
        acc = float((psnrs_t > 20).sum().item() / max(1, psnrs_t.numel())) if psnrs_t.numel() else 0.0
        qtls = (torch.quantile(psnrs_t, QUANTILES, interpolation="lower").tolist()
                if psnrs_t.numel() else [0, 0, 0, 0, 0])
        qtls_str = "[" + ", ".join(f"{q:.3f}" for q in qtls) + "]"
        psnr_mean = psnrs_t.mean().item() if psnrs_t.numel() else 0.0

        return (
            loss_accum / size_count,
            acc,
            rpd_sum / size_count,
            polar_sum / size_count,
            qtls_str,
            psnr_mean,
            inv_lap_sum / size_count,
            l1_sum / size_count,
            l2_sum / size_count,
        )

    @torch.no_grad()
    def eval_epoch(loader):
        model.eval()
        size_count = 0
        loss_sum = 0.0
        rpd_sum = 0.0
        polar_sum = 0.0
        psnr_list = []
        inv_lap_sum = 0.0
        l1_sum = 0.0
        l2_sum = 0.0

        for data in loader:
            data = data.to(device, non_blocking=True)
            out = forward_for_task(data)
            n = out.size(0)

            if getattr(data, "y", None) is None:
                print("[warn] Batch has no 'y' labels; skipping (flow-only).")
                continue

            target = torch.reshape(data.y, (n, -1)).to(device)
            loss = criterion(out, target)
            loss_sum += loss.item() * n
            size_count += n

            try:
                psnr_list.append(PSNR(out, target))
                rpd_sum += RPD(out, target).sum().item()
                polar_sum += polar_loss(out, target, lmbda=1).sum().item()
                inv_lap_sum += inv_lap(out, target).sum().item()
                l1_sum += F.l1_loss(out, target, reduction="sum").item()
                l2_sum += l2_error(out, target).sum().item()
            except Exception:
                pass

        if size_count == 0:
            return (float('nan'), 0.0, float('nan'), float('nan'), "[n/a]",
                    float('nan'), float('nan'), float('nan'), float('nan'))

        psnrs_t = torch.cat(psnr_list) if len(psnr_list) else torch.tensor([], device=device)
        acc = float((psnrs_t > 20).sum().item() / max(1, psnrs_t.numel())) if psnrs_t.numel() else 0.0
        qtls = (torch.quantile(psnrs_t, QUANTILES, interpolation="lower").tolist()
                if psnrs_t.numel() else [0, 0, 0, 0, 0])
        qtls_str = "[" + ", ".join(f"{q:.3f}" for q in qtls) + "]"
        psnr_mean = psnrs_t.mean().item() if psnrs_t.numel() else 0.0

        return (
            loss_sum / size_count,
            acc,
            rpd_sum / size_count,
            polar_sum / size_count,
            qtls_str,
            psnr_mean,
            inv_lap_sum / size_count,
            l1_sum / size_count,
            l2_sum / size_count,
        )

    print("Begin training...")
    best_val_acc = -1.0
    best_state = None

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    with open(metrics_csv, "w") as f:
        f.write("epoch,split,loss,acc,rpd,polar,psnr_qtls,psnr_mean,inv_lap,l1,l2\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        tr = train_epoch(train_loader)
        va = eval_epoch(val_loader)

        # unpack
        (train_loss, train_acc, train_rpd, train_pl, train_qtls, train_psnr,
         train_inv_lap, train_l1, train_l2) = tr
        (val_loss, val_acc, val_rpd, val_pl, val_qtls, val_psnr,
         val_inv_lap, val_l1, val_l2) = va

        # write metrics
        with open(metrics_csv, "a") as f:
            f.write(f"{epoch},train,{train_loss:.6e},{train_acc:.6f},{train_rpd:.6e},{train_pl:.6e},"
                    f"\"{train_qtls}\",{train_psnr:.6f},{train_inv_lap:.6e},{train_l1:.6e},{train_l2:.6e}\n")
            f.write(f"{epoch},val,{val_loss:.6e},{val_acc:.6f},{val_rpd:.6e},{val_pl:.6e},"
                    f"\"{val_qtls}\",{val_psnr:.6f},{val_inv_lap:.6e},{val_l1:.6e},{val_l2:.6e}\n")

        # track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        # step scheduler
        scheduler1.step()
        epoch_end = time.time()
        print(
            f"Epoch {epoch:03d}: train loss {train_loss:.3e}, val loss {val_loss:.3e}, "
            f"train acc {train_acc*100:6.3f}, val acc {val_acc*100:6.3f} "
            f"({epoch_end - epoch_start:.1f}s)"
        )
        sys.stdout.flush()

        # periodic saves
        if epoch % 20 == 0:
            if best_state is not None:
                torch.save(best_state, os.path.join(run_dir, "best.abs.pt"))
            torch.save(model.state_dict(), os.path.join(run_dir, f"model-{epoch:03d}.pt"))

    # final saves
    if best_state is not None:
        torch.save(best_state, os.path.join(run_dir, "best.abs.pt"))
    torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pt"))

    # test
    test_metrics = eval_epoch(test_loader)
    (test_loss, test_acc, test_rpd, test_pl, test_qtls, test_psnr,
     test_inv_lap, test_l1, test_l2) = test_metrics

    report = {
        "loss": float(test_loss),
        "acc": float(test_acc),
        "rpd": float(test_rpd),
        "polar": float(test_pl),
        "psnr_quantiles": test_qtls,
        "psnr_mean": float(test_psnr),
        "inv_lap": float(test_inv_lap),
        "l1": float(test_l1),
        "l2": float(test_l2),
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"Test loss: {test_loss:.3e}, test acc: {test_acc*100:.3f}, test RPD: {test_rpd:.3e}, "
        f"test pl: {test_pl:.3e}, test quantiles: {test_qtls}, test inv lap: {test_inv_lap:.3e}, "
        f"test l1 loss: {test_l1:.3e}, test l2 loss: {test_l2:.3e}"
    )
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("The job finished at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="TrainEGNN", description="Train EGNN on mesh graphs")
    parser.add_argument("--workspace", type=str, default="/scratch/bdbq/mcampos1/eigenEGNN_model")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--datasets", nargs="+", default=["test"], choices=["Synth", "Thingi10k", "Tori", "SHREC", "test"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", dest="loss_fn", default="L2", choices=["RPD", "Polar", "Polar2", "L1", "L2"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--scheduler", default="Cos", choices=["Cos", "Exp"])
    parser.add_argument("--hidden", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--mlp-hidden", dest="mlp_hidden", type=int, default=64)
    parser.add_argument("--mlp-layers", dest="mlp_layers", type=int, default=5)
    args = parser.parse_args()

    run(
        workspace=args.workspace,
        output_dir=args.output_dir,
        batch_size=args.batch,
        datasets=tuple(args.datasets),
        scheduler=args.scheduler,
        loss_fn=args.loss_fn,
        lr=args.lr,
        epochs=args.epochs,
        model_type="EGNN",
        hidden_channels=tuple(args.hidden),
        MLP_hidden_channels=args.mlp_hidden,
        MLP_num_layers=args.mlp_layers,
    )
