
import numpy as np, bm3d
from PIL import Image
from pathlib import Path

def denoise_file(args):
    src, dst, sigma = args
    if Path(dst).exists():
        return dst
    a = np.asarray(Image.open(src).convert("RGB"))[:, :560].mean(-1).astype(np.uint8)
    x = np.log1p(a.astype(np.float64)) / np.log(256.0)
    out = bm3d.bm3d(x, sigma_psd=float(sigma))
    back = np.expm1(np.clip(out, 0, 1) * np.log(256.0))
    Image.fromarray(np.clip(back, 0, 255).astype(np.uint8)).save(dst)
    return dst
