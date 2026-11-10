"""CROHME dataset loader.

Expects the CoMER/BTTR layout (data.zip unzipped):
    data/<split>/caption.txt   lines: "<name>\t<tok tok tok>"  (or space after name)
    data/<split>/img/<name>.bmp
splits: train, 2014, 2016, 2019
"""
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MAX_H, MAX_W = 256, 1024


def read_captions(split_dir):
    items = []
    with open(os.path.join(split_dir, "caption.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items.append((parts[0], parts[1:]))
    return items


def find_image(split_dir, name):
    for ext in (".bmp", ".png", ".jpg"):
        p = os.path.join(split_dir, "img", name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def preprocess(img: Image.Image, augment=False):
    """-> float tensor [1,H,W], ink=1 on background=0."""
    img = img.convert("L")
    if augment:
        s = random.uniform(0.7, 1.4)
        img = img.resize((max(8, int(img.width * s)), max(8, int(img.height * s))), Image.BILINEAR)
        img = img.rotate(random.uniform(-4, 4), resample=Image.BILINEAR, expand=True,
                         fillcolor=int(np.median(np.asarray(img))))
    x = np.asarray(img, dtype=np.float32) / 255.0
    if x.mean() > 0.5:          # white background -> invert
        x = 1.0 - x
    # crop to ink + small margin
    ys, xs = np.where(x > 0.2)
    if len(ys):
        m = 4
        x = x[max(0, ys.min() - m): ys.max() + m + 1, max(0, xs.min() - m): xs.max() + m + 1]
    h, w = x.shape
    scale = min(1.0, MAX_H / h, MAX_W / w)
    if scale < 1.0:
        x = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize(
            (max(16, int(w * scale)), max(16, int(h * scale))), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0)


class CROHME(Dataset):
    def __init__(self, root, split, vocab, augment=False):
        self.dir = os.path.join(root, split)
        self.items = read_captions(self.dir)
        self.vocab, self.augment = vocab, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, toks = self.items[i]
        img = preprocess(Image.open(find_image(self.dir, name)), self.augment)
        return img, torch.tensor(self.vocab.encode(toks)), name


def collate(batch):
    imgs, seqs, names = zip(*batch)
    H = max(i.shape[1] for i in imgs)
    W = max(i.shape[2] for i in imgs)
    B = len(imgs)
    x = torch.zeros(B, 1, H, W)
    mask = torch.ones(B, H, W, dtype=torch.bool)   # True = padding
    for k, im in enumerate(imgs):
        x[k, :, : im.shape[1], : im.shape[2]] = im
        mask[k, : im.shape[1], : im.shape[2]] = False
    L = max(len(s) for s in seqs)
    y = torch.zeros(B, L, dtype=torch.long)        # pad id = 0
    for k, s in enumerate(seqs):
        y[k, : len(s)] = s
    return x, mask, y, list(names)
