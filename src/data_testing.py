import torch
from torch_geometric.data import Data
from egnn import EGNN
from train import safe_load_pt

graphs = safe_load_pt("/scratch/bdbq/mcampos1/eigenEGNN_model/data_processed/processed_test.pt")

g = graphs[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EGNN(
    input_channels=g.x.size(1) if g.x is not None else 1,
    output_channels=g.y.numel() if g.y.ndim==1 else g.y.size(-1),
    hidden_channels=[32,64,128],
    MLP_hidden_channels=64,
    MLP_num_layers=2
).to(device)

out = model(g.x.to(device), g.edge_index.to(device), torch.zeros(g.x.size(0), dtype=torch.long, device=device), pos=g.pos.to(device))
print("\n", out)
print(out.mean(), out.std())

print("\n ################# \n")

ys = [g.y for g in graphs if hasattr(g, "y")]
ys_t = torch.cat(ys, dim=0).float()

print("y shape:", ys_t.shape)
print("y mean:", ys_t.mean().item())
print("y std:", ys_t.std().item())
print("y min/max:", ys_t.min().item(), ys_t.max().item())

print("Example y[0]:", ys_t[0])