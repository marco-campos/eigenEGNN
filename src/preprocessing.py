# preprocess.py
import os
import argparse
from pathlib import Path
from typing import List, Optional, Union, Dict

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ------------------------------
# OBJ parsing & graph building
# ------------------------------

def load_obj_vertices_faces(path: Path):
    """
    Minimal, robust OBJ loader:
      - Parses vertex positions 'v x y z'
      - Parses faces 'f i j k [l ...]' (supports 'i/j/k' tokens)
      - Triangulates n-gons to triangles
    Returns:
      V: np.ndarray [N, 3] float32
      F: np.ndarray [T, 3] int64 (0-based vertex indices)
    """
    verts = []
    tris = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):  # vertex position
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = parts[1:4]
                    verts.append([float(x), float(y), float(z)])
            elif line.startswith("f "):  # face indices (1-based)
                toks = line.strip().split()[1:]
                # Keep only the vertex index before any slash.
                idx = [int(tok.split("/")[0]) - 1 for tok in toks]
                if len(idx) < 3:
                    continue
                # Fan triangulation if polygon
                for k in range(1, len(idx) - 1):
                    tris.append([idx[0], idx[k], idx[k + 1]])

    V = np.asarray(verts, dtype=np.float32)
    F = np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), dtype=np.int64)
    return V, F


def faces_to_undirected_edges(F: np.ndarray, num_nodes: int) -> torch.Tensor:
    """
    Collect unique undirected edges from triangle faces and return directed
    edges for message passing (u->v and v->u).
    """
    if F.size == 0:
        return torch.empty((2, 0), dtype=torch.long)

    E = set()
    for a, b, c in F:
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            E.add(e)

    undirected = []
    for u, v in E:
        undirected.append([u, v])
        undirected.append([v, u])

    edge_index = torch.tensor(undirected, dtype=torch.long).t().contiguous()
    return edge_index


# ------------------------------
# Targets (eigenvalues) loading
# ------------------------------

def load_eigs_txt(txt_path: Path, num_eigs: int) -> Optional[torch.Tensor]:
    """
    Load the first num_eigs lines from a .txt file as float32 tensor [num_eigs].
    Returns None if the file is unreadable or has fewer than num_eigs lines.
    """
    try:
        vals: List[float] = []
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    vals.append(float(line))
                except ValueError:
                    return None  # malformed line
                if len(vals) >= num_eigs:
                    break
        if len(vals) < num_eigs:
            return None
        return torch.tensor(vals, dtype=torch.float32)
    except Exception:
        return None


def index_eigs_dir(eigs_dir: Path) -> Dict[str, Path]:
    """
    Build a stem -> path index for fast matching of foo.obj to eigs/foo.txt.
    """
    mapping: Dict[str, Path] = {}
    for p in eigs_dir.rglob("*.txt"):
        mapping[p.stem] = p
    return mapping


def mesh_to_data(
    mesh_path: Path,
    eigs_index: Optional[Dict[str, Path]] = None,
    num_eigs: int = 10,
    add_constant_feature: bool = True,
    strict_targets: bool = True,
) -> Optional[Data]:
    """
    Convert a single OBJ mesh into a torch_geometric Data:
      - pos: [N, 3]
      - x:   [N, 1] (ones) if add_constant_feature else None
      - edge_index: [2, E]
      - y: [num_eigs] if eigs_index is provided and a matching .txt exists;
           None if missing and strict_targets=False; otherwise returns None to skip.
    """
    V, F = load_obj_vertices_faces(mesh_path)
    if V.shape[0] == 0:
        print(f"[warn] No vertices in OBJ: {mesh_path}")
        return None

    pos = torch.from_numpy(V)  # [N,3]
    N = pos.size(0)
    x = torch.ones((N, 1), dtype=torch.float32) if add_constant_feature else None
    edge_index = faces_to_undirected_edges(F, N)

    y: Optional[torch.Tensor] = None
    if eigs_index is not None:
        stem = mesh_path.stem
        txt_path = eigs_index.get(stem, None)
        if txt_path is not None:
            y = load_eigs_txt(txt_path, num_eigs=num_eigs)
        if y is None and strict_targets:
            # Missing or malformed target → skip this mesh
            print(f"[warn] Missing/invalid eigs for {mesh_path.name} (expected in eigs/ as {stem}.txt); skipping.")
            return None

    data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
    data.mesh_path = str(mesh_path)
    return data


# ------------------------------
# Dataset builder
# ------------------------------

def build_mesh_dataset(
    meshes_dir: Union[str, Path],
    eigs_dir: Optional[Union[str, Path]] = None,
    num_eigs: int = 10,
    limit: Optional[int] = None,
    shuffle: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    allow_missing_targets: bool = False,
) -> List[Data]:
    """
    Scan a directory for .obj files (recursively), convert each to a Data graph.
    Also attaches y targets from eigs_dir if provided (or inferred).
    Args:
      meshes_dir: folder with .obj meshes (e.g., ../../data/processed_Tori/meshes)
      eigs_dir:   folder with .txt targets (e.g., ../../data/processed_Tori/eigs)
                  If None, inferred as sibling of meshes_dir's parent.
      num_eigs:   number of lines/values to read per .txt
      limit:      optional cap on number of meshes (for quick tests)
      shuffle:    randomize order
      save_path:  optional .pt path to save list[Data]
      allow_missing_targets: if True, keep graphs with y=None; otherwise skip them
    Returns:
      data_list: list[Data]
    """
    meshes_dir = Path(meshes_dir)
    if not meshes_dir.exists():
        raise FileNotFoundError(f"meshes_dir not found: {meshes_dir}")

    # Infer eigs_dir if not given: assume sibling "eigs" under dataset root
    if eigs_dir is None:
        dataset_root = meshes_dir.parent  # processed_* folder
        eigs_dir = dataset_root / "eigs"
    eigs_dir = Path(eigs_dir)

    eigs_index = None
    if eigs_dir.exists():
        eigs_index = index_eigs_dir(eigs_dir)
        if len(eigs_index) == 0:
            print(f"[warn] eigs_dir exists but no .txt files found: {eigs_dir}")
    else:
        print(f"[warn] eigs_dir not found: {eigs_dir} (proceeding without targets)")
        allow_missing_targets = True  # no targets available at all

    mesh_paths = list(meshes_dir.rglob("*.obj"))
    if len(mesh_paths) == 0:
        raise FileNotFoundError(f"No .obj files found under {meshes_dir}")

    if shuffle:
        rng = np.random.default_rng(0xC0FFEE)
        rng.shuffle(mesh_paths)
    if limit is not None and limit > 0:
        mesh_paths = mesh_paths[:limit]

    data_list: List[Data] = []
    for p in tqdm(mesh_paths, desc="Building graphs"):
        data = mesh_to_data(
            p,
            eigs_index=eigs_index,
            num_eigs=num_eigs,
            add_constant_feature=True,
            strict_targets=not allow_missing_targets,
        )
        if data is not None:
            data_list.append(data)

    kept = len(data_list)
    total = len(mesh_paths)
    print(f"[info] Built {kept}/{total} graphs "
          f"({'with/without' if allow_missing_targets else 'with'} targets).")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_list, save_path)
        print(f"[info] Saved {len(data_list)} graphs to {save_path}")

    return data_list


def _print_batch_summary(batch):
    print("Batch summary:")
    print(f"  nodes: {batch.num_nodes}")
    print(f"  edges: {batch.num_edges} (directed)")
    print(f"  graphs in batch: {batch.num_graphs}")
    print(f"  x shape: {tuple(batch.x.shape)}")
    print(f"  pos shape: {tuple(batch.pos.shape)}")
    print(f"  edge_index shape: {tuple(batch.edge_index.shape)}")
    if getattr(batch, "y", None) is not None:
        print(f"  y shape: {tuple(batch.y.shape)}")
    else:
        print("  y: None")


if __name__ == "__main__":
    # Example paths aligned with your tree:
    #   ../../../data/processed_Tori/meshes
    #   ../../../data/processed_Tori/eigs
    parser = argparse.ArgumentParser(prog="preprocess", description="preprocess data into torchgeometric datasets")
    parser.add_argument("--data", dest="data", default="Tori", type=str)
    parser.add_argument("--limit", dest="limit", default=-1, type=int)
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--num-eigs", dest="num_eigs", default=10, type=int)
    parser.add_argument("--allow-missing", dest="allow_missing", action="store_true",
                        help="Keep graphs even if y targets are missing/malformed")

    args = parser.parse_args()

    meshes_dir = f"../../../data/processed_{args.data}/meshes"
    eigs_dir   = f"../../../data/processed_{args.data}/eigs"
    save_path  = f"../data_processed/processed_{args.data}.pt"

    data_list = build_mesh_dataset(
        meshes_dir=meshes_dir,
        eigs_dir=eigs_dir,
        num_eigs=args.num_eigs,
        shuffle=True,
        limit=None if args.limit == -1 else args.limit,
        save_path=save_path,
        allow_missing_targets=args.allow_missing,
    )

    if args.test:
        loader = DataLoader(data_list, batch_size=2, shuffle=False)
        for i, batch in enumerate(loader):
            print(f"\n--- Mini-batch {i} ---")
            _print_batch_summary(batch)
            # Example: logits = model(batch.x, batch.edge_index, batch.batch, pos=batch.pos)
