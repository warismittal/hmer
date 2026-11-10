"""LaTeX token vocabulary. CROHME captions are already space-separated tokens."""
import json

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


class Vocab:
    def __init__(self, tokens):
        self.itos = [PAD, SOS, EOS, UNK] + sorted(set(tokens) - {PAD, SOS, EOS, UNK})
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    pad, sos, eos, unk = 0, 1, 2, 3

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.sos] + [self.stoi.get(t, self.unk) for t in tokens] + [self.eos]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i in (self.pad, self.sos):
                continue
            out.append(self.itos[i])
        return out

    def save(self, path):
        json.dump(self.itos, open(path, "w"))

    @classmethod
    def load(cls, path):
        v = cls([])
        v.itos = json.load(open(path))
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v
