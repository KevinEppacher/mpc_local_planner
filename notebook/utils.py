
import math, time, sys, os, random
import numpy as np
import torch

# --- Virtual costmap (two Gaussians) ---
def make_gaussian_costmap(W, H, res, peaks):
    # peaks = list of (x[m], y[m], sigma[m], amplitude[0..1])
    xs = np.arange(W) * res
    ys = np.arange(H) * res
    X, Y = np.meshgrid(xs, ys)
    G = np.zeros((H, W), dtype=np.float32)
    for (px, py, s, a) in peaks:
        G += a * np.exp(-((X - px)**2 + (Y - py)**2) / (2 * s*s))
    G = np.clip(G, 0.0, 1.0)
    return torch.from_numpy(G)