"""python evaluate.py --ckpt runs/base/best.pt --data data --splits 2014 2016 2019 [--beam 10]"""
import argparse
import torch
from tqdm import tqdm
from data import CROHME, collate
from model import HMERModel
from vocab import Vocab
from metrics import exprate

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data", default="data")
ap.add_argument("--splits", nargs="+", default=["2014", "2016", "2019"])
ap.add_argument("--beam", type=int, default=10)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
s = torch.load(a.ckpt, map_location=dev)
vocab = Vocab([]); vocab.itos = s["vocab"]; vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
model = HMERModel(len(vocab)).to(dev).eval(); model.load_state_dict(s["model"])

for sp in a.splits:
    ds = CROHME(a.data, sp, vocab)
    preds, gts = [], []
    for i in tqdm(range(len(ds)), desc=sp):
        x, y, _ = ds[i]
        x, m, _, _ = collate([(x, y, "")])
        out = model.beam(x.to(dev), m.to(dev), a.beam)
        preds.append(vocab.decode(out)); gts.append(vocab.decode(y.tolist()))
    r = exprate(preds, gts)
    print(f"{sp}: ExpRate {r['exprate']:.2f} | <=1 {r['le1']:.2f} | <=2 {r['le2']:.2f}")
