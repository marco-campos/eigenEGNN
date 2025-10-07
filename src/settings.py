import torch

NUM_EIGS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
