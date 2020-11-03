import torch
from utils import *

def solve(Ax, b, iters=10, tolerance=1e-10):

    x = torch.zeros_like(b)
    r = b - Ax(x)
    p = r.clone()

    for i in range(iters):
        r2 = r.dot(r)
        Ap = Ax(p)
        α = r2 / p.dot(Ap)
        x += α * p
        r -= α * Ap
        r2_old, r2 = r2, r.dot(r)
        β = r2/r2_old
        p = r + β * p
        if r2 < tolerance: break

    return x
