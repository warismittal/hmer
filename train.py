"""Train HMER on CROHME.  Usage:
    python train.py --data data --out runs/base --epochs 200
    python train.py --data data --out runs/base --resume      # continue (e.g. new Kaggle session)
"""
import argparse, math, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CROHME, collate, read_captions
from model import HMERModel
from vocab import Vocab
from metrics import exprate


def build_vocab(root, out):
    path = os.path.join(out, "vocab.json")
    if os.path.exists(path):
        return Vocab.load(path)
    toks = [t for _, seq in read_captions(os.path.join(root, "train")) for t in seq]
    v = Vocab(toks)
    v.save(path)
    return v


def evaluate(model, loader, vocab, device, limit=None):
    model.eval()
    preds, gts = [], []
    for n, (x, m, y, _) in enumerate(loader):
        x, m = x.to(device), m.to(device)
        out = model.greedy(x, m)
        preds += [vocab.decode(o.tolist()) for o in out]
        gts += [vocab.decode(t.tolist()) for t in y]
        if limit and n + 1 >= limit:
            break
    model.train()
    return exprate(preds, gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--val", default="2014")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time_limit_h", type=float, default=11.0, help="stop safely before Kaggle's 12h cap")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    vocab = build_vocab(a.data, a.out)
    print(f"device={dev} vocab={len(vocab)}")

    tr = DataLoader(CROHME(a.data, "train", vocab, augment=True), a.bs, shuffle=True,
                    collate_fn=collate, num_workers=a.workers, drop_last=True)
    va = DataLoader(CROHME(a.data, a.val, vocab), a.bs, collate_fn=collate, num_workers=a.workers)

    model = HMERModel(len(vocab)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    start, best = 0, 0.0
    ck = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); start, best = s["epoch"] + 1, s["best"]
        print(f"resumed from epoch {start}, best ExpRate {best:.2f}")

    t0 = time.time()
    for ep in range(start, a.epochs):
        pbar, tot = tqdm(tr, desc=f"ep {ep}"), 0.0
        for x, m, y, _ in pbar:
            x, m, y = x.to(dev), m.to(dev), y.to(dev)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                logits = model(x, m, y[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y[:, 1:].reshape(-1),
                                       ignore_index=0, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            r = evaluate(model, va, vocab, dev)
            print(f"ep {ep} loss {tot/len(tr):.3f} | {a.val}: ExpRate {r['exprate']:.2f} "
                  f"<=1 {r['le1']:.2f} <=2 {r['le2']:.2f}")
            if r["exprate"] > best:
                best = r["exprate"]
                torch.save({"model": model.state_dict(), "vocab": vocab.itos}, os.path.join(a.out, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best}, ck)
        if (time.time() - t0) / 3600 > a.time_limit_h:
            print("time limit reached, checkpoint saved. rerun with --resume"); break


if __name__ == "__main__":
    main()
