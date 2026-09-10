"""Dataset, features and length-bucketed batching for Shona CTC training."""
import json, os, random, sys
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import LogMel, spec_augment

SR = 16000

class Vocab:
    def __init__(self, path="text/vocab.json"):
        d = json.load(open(path, encoding="utf-8"))
        self.itos = d["vocab"]
        self.blank = d["blank_id"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i != self.blank)


class ShonaASR(Dataset):
    """Reads the JSONL manifests produced by scripts/extract.py."""

    def __init__(self, manifest, vocab, n_mels=80, max_dur=None, min_dur=0.5,
                 train=False, spec_augment=True):
        self.rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
        self.rows = [r for r in self.rows
                     if r["duration"] >= min_dur
                     and (max_dur is None or r["duration"] <= max_dur)]
        self.rows.sort(key=lambda r: r["duration"])
        self.vocab = vocab
        self.train = train
        self.spec_augment = spec_augment and train
        self.mel = LogMel(SR, n_fft=400, hop=160, n_mels=n_mels)

    def __len__(self):
        return len(self.rows)

    def durations(self):
        return [r["duration"] for r in self.rows]

    def __getitem__(self, i):
        r = self.rows[i]
        wav, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
        assert sr == SR, f"{r['audio']} is {sr} Hz, expected {SR}"
        wav = torch.from_numpy(wav.mean(1))
        feat = self.mel(wav)                                    # (n_mels, T)
        if self.spec_augment:
            feat = spec_augment(feat)
        tgt = torch.tensor(self.vocab.encode(r["text"]), dtype=torch.long)
        return feat, tgt, r["id"]


def collate(batch):
    feats, tgts, ids = zip(*batch)
    fl = torch.tensor([f.shape[1] for f in feats], dtype=torch.long)
    tl = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    n_mels, T = feats[0].shape[0], int(fl.max())
    x = torch.zeros(len(feats), n_mels, T)
    for i, f in enumerate(feats):
        x[i, :, : f.shape[1]] = f
    return x, fl, torch.cat(tgts), tl, ids


class FrameBudgetSampler(Sampler):
    """Batches of roughly equal total frames -- clips here average ~20 s, so a
    fixed batch size would blow up memory on the long tail."""

    def __init__(self, dataset, max_frames=1_600_000, max_batch=64,
                 shuffle=True, seed=0):
        self.durs = dataset.durations()
        self.max_frames = max_frames
        self.max_batch = max_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.batches = self._build()

    def _build(self):
        batches, cur, longest = [], [], 0
        for i, d in enumerate(self.durs):          # dataset is duration-sorted
            frames = int(d * SR / 160)
            longest = max(longest, frames)
            if cur and (longest * (len(cur) + 1) > self.max_frames
                        or len(cur) >= self.max_batch):
                batches.append(cur)
                cur, longest = [], frames
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        b = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(b)
        return iter(b)

    def __len__(self):
        return len(self.batches)
