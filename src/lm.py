"""Character-level LSTM language model, for shallow fusion during decoding."""
import json, math, torch
import torch.nn as nn


class CharLM(nn.Module):
    def __init__(self, n_classes, emb=128, hidden=512, layers=2, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb)
        self.rnn = nn.LSTM(emb, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x, state=None):
        y, state = self.rnn(self.emb(x), state)
        return self.out(self.drop(y)), state

    @torch.no_grad()
    def score_next(self, x, state=None):
        """Log-probs over the next character given a prefix step."""
        logits, state = self.forward(x, state)
        return logits[:, -1].log_softmax(-1), state


def build_corpus(path, vocab, bos_id):
    """Flatten the text corpus into one long id stream, newline -> bos."""
    ids = []
    for line in open(path, encoding="utf-8"):
        ids.append(bos_id)
        ids.extend(vocab.encode(line.strip()))
    return torch.tensor(ids, dtype=torch.long)


def train_lm(train_txt, valid_txt, vocab_path="text/vocab.json", epochs=10,
             bptt=200, batch=64, lr=2e-3, device=None, out=None):
    import os
    from data import Vocab
    if out is None:
        # Name the checkpoint after the corpus so the Protocol A and B LMs
        # cannot silently overwrite each other.
        tag = "spk" if "spk" in os.path.basename(train_txt) else "official"
        out = f"checkpoints/lm_{tag}.pt"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab(vocab_path)
    bos = vocab.blank                      # reuse blank slot as sequence start
    model = CharLM(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()

    def batchify(t):
        n = t.numel() // batch
        return t[: n * batch].view(batch, n).to(device)

    tr = batchify(build_corpus(train_txt, vocab, bos))
    va = batchify(build_corpus(valid_txt, vocab, bos))

    def evaluate(d):
        model.eval(); tot = k = 0
        with torch.no_grad():
            for i in range(0, d.size(1) - 1, bptt):
                x = d[:, i : i + bptt]
                y = d[:, i + 1 : i + 1 + x.size(1)]
                if y.size(1) < x.size(1):
                    x = x[:, : y.size(1)]
                logits, _ = model(x)
                tot += lossf(logits.reshape(-1, len(vocab)), y.reshape(-1)).item() * y.numel()
                k += y.numel()
        return math.exp(tot / max(k, 1))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train(); state = None
        for i in range(0, tr.size(1) - 1, bptt):
            x = tr[:, i : i + bptt]
            y = tr[:, i + 1 : i + 1 + x.size(1)]
            if y.size(1) < x.size(1):
                x = x[:, : y.size(1)]
            logits, state = model(x, state)
            state = tuple(s.detach() for s in state)
            loss = lossf(logits.reshape(-1, len(vocab)), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ppl = evaluate(va)
        print(f"lm epoch {ep}: valid char-ppl {ppl:.3f}", flush=True)
        if ppl < best:
            best = ppl
            torch.save({"model": model.state_dict(), "vocab": vocab.itos,
                        "train_txt": train_txt, "valid_ppl": ppl}, out)
    print(f"best valid char-ppl {best:.3f} -> {out}")


if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="text/lm_train.txt",
                    help="Protocol A: text/lm_train.txt. "
                         "Protocol B: text/lm_spk_train.txt -- using the "
                         "Protocol A corpus to decode spk_test leaks every "
                         "one of its transcripts into the LM.")
    ap.add_argument("--valid", default="text/lm_valid.txt")
    ap.add_argument("--out", default=None,
                    help="default: checkpoints/lm_official.pt or lm_spk.pt, "
                         "named after the training corpus")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bptt", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    train_lm(a.train, a.valid, epochs=a.epochs, bptt=a.bptt, batch=a.batch,
             lr=a.lr, out=a.out)
