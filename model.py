"""CNN (DenseNet-B) encoder + Transformer decoder for HMER."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Encoder ----------------
class Bottleneck(nn.Module):
    def __init__(self, cin, growth, drop=0.2):
        super().__init__()
        self.bn1, self.conv1 = nn.BatchNorm2d(cin), nn.Conv2d(cin, 4 * growth, 1, bias=False)
        self.bn2, self.conv2 = nn.BatchNorm2d(4 * growth), nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.drop(self.conv2(F.relu(self.bn2(out))))
        return torch.cat([x, out], 1)


def dense_block(cin, n, growth):
    layers = []
    for _ in range(n):
        layers.append(Bottleneck(cin, growth))
        cin += growth
    return nn.Sequential(*layers), cin


class Transition(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.bn, self.conv = nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return F.avg_pool2d(self.conv(F.relu(self.bn(x))), 2, ceil_mode=True)


class DenseNetEncoder(nn.Module):
    """Total downsampling = 16."""
    def __init__(self, growth=24, depth=16, d_model=256):
        super().__init__()
        c = 2 * growth
        self.stem = nn.Sequential(nn.Conv2d(1, c, 7, 2, 3, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
                                  nn.MaxPool2d(2, ceil_mode=True))
        self.b1, c = dense_block(c, depth, growth)
        self.t1 = Transition(c, c // 2); c //= 2
        self.b2, c = dense_block(c, depth, growth)
        self.t2 = Transition(c, c // 2); c //= 2
        self.b3, c = dense_block(c, depth, growth)
        self.norm = nn.BatchNorm2d(c)
        self.proj = nn.Conv2d(c, d_model, 1)

    def forward(self, x, mask):
        f = self.stem(x)
        f = self.t1(self.b1(f))
        f = self.t2(self.b2(f))
        f = self.proj(F.relu(self.norm(self.b3(f))))          # [B,D,h,w]
        h, w = f.shape[-2:]
        m = F.interpolate(mask[:, None].float(), size=(h, w), mode="nearest")[:, 0].bool()
        return f, m


# ---------------- Positional encodings ----------------
class PosEnc2D(nn.Module):
    """Sinusoidal 2D encoding using normalized coordinates (robust to image size)."""
    def __init__(self, d_model):
        super().__init__()
        self.d = d_model // 2

    def forward(self, f, mask):
        B, D, h, w = f.shape
        valid = (~mask).float()
        y = valid.cumsum(1) / (valid.sum(1, keepdim=True).clamp(min=1))
        x = valid.cumsum(2) / (valid.sum(2, keepdim=True).clamp(min=1))
        div = torch.exp(torch.arange(0, self.d, 2, device=f.device).float() * (-math.log(10000.0) / self.d))
        def enc(c):
            c = c[..., None] * 2 * math.pi * 10 * div    # [B,h,w,d/2]
            return torch.cat([c.sin(), c.cos()], -1)
        pe = torch.cat([enc(y), enc(x)], -1).permute(0, 3, 1, 2)   # [B,D,h,w]
        return f + pe


class PosEnc1D(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len)[:, None].float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


# ---------------- Full model ----------------
class HMERModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, layers=3, ff=1024, dropout=0.3):
        super().__init__()
        self.encoder = DenseNetEncoder(d_model=d_model)
        self.pos2d = PosEnc2D(d_model)
        self.enc_norm = nn.LayerNorm(d_model)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos1d = PosEnc1D(d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def encode(self, img, mask):
        f, m = self.encoder(img, mask)
        f = self.pos2d(f, m)
        mem = self.enc_norm(f.flatten(2).transpose(1, 2))     # [B,hw,D]
        return mem, m.flatten(1)

    def decode(self, tgt, mem, mem_mask):
        L = tgt.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt.device), 1)
        x = self.pos1d(self.embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, mem, tgt_mask=causal, tgt_key_padding_mask=(tgt == 0),
                         memory_key_padding_mask=mem_mask)
        return self.out(x)

    def forward(self, img, mask, tgt_in):
        mem, mm = self.encode(img, mask)
        return self.decode(tgt_in, mem, mm)

    @torch.no_grad()
    def greedy(self, img, mask, sos=1, eos=2, max_len=200):
        mem, mm = self.encode(img, mask)
        B = img.size(0)
        ys = torch.full((B, 1), sos, dtype=torch.long, device=img.device)
        done = torch.zeros(B, dtype=torch.bool, device=img.device)
        for _ in range(max_len):
            nxt = self.decode(ys, mem, mm)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, 0)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        return ys[:, 1:]

    @torch.no_grad()
    def beam(self, img, mask, beam=10, sos=1, eos=2, max_len=200, alpha=1.0):
        """Beam search for a single image (B=1). Length-normalized scores."""
        mem, mm = self.encode(img, mask)
        beams = [([sos], 0.0)]
        finished = []
        for _ in range(max_len):
            seqs = torch.tensor([b[0] for b in beams], device=img.device)
            k = len(beams)
            logp = self.decode(seqs, mem.expand(k, -1, -1), mm.expand(k, -1))[:, -1].log_softmax(-1)
            cand = []
            for i, (s, sc) in enumerate(beams):
                top = logp[i].topk(beam)
                for lp, t in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((s + [t], sc + lp))
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for s, sc in cand:
                if s[-1] == eos:
                    finished.append((s, sc / (len(s) - 1) ** alpha))
                else:
                    beams.append((s, sc))
                if len(beams) == beam:
                    break
            if len(finished) >= beam or not beams:
                break
        if not finished:
            finished = [(s, sc / len(s) ** alpha) for s, sc in beams]
        best = max(finished, key=lambda f: f[1])[0]
        return best[1:]
