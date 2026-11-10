"""python predict.py --ckpt runs/base/best.pt image.png"""
import argparse
import torch
from PIL import Image
from data import preprocess, collate
from model import HMERModel
from vocab import Vocab


def load(ckpt, dev="cpu"):
    s = torch.load(ckpt, map_location=dev)
    v = Vocab([]); v.itos = s["vocab"]; v.stoi = {t: i for i, t in enumerate(v.itos)}
    m = HMERModel(len(v)).to(dev).eval(); m.load_state_dict(s["model"])
    return m, v


def predict(model, vocab, img, beam=10, dev="cpu"):
    x = preprocess(img)
    x, m, _, _ = collate([(x, torch.tensor([1, 2]), "")])
    return " ".join(vocab.decode(model.beam(x.to(dev), m.to(dev), beam)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("image")
    a = ap.parse_args()
    model, vocab = load(a.ckpt)
    print(predict(model, vocab, Image.open(a.image)))
