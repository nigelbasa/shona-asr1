"""Train the CNN-CTC Shona acoustic model."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ShonaASR, Vocab, collate, FrameBudgetSampler
from model import ShonaCNN
from decode import greedy_decode, wer, cer


def build_loader(manifest, vocab, args, train):
    ds = ShonaASR(manifest, vocab, n_mels=args.n_mels,
                  max_dur=args.max_dur if train else None, train=train)
    sampler = FrameBudgetSampler(ds, max_frames=args.max_frames,
                                 max_batch=args.max_batch, shuffle=train)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                      num_workers=args.workers, pin_memory=True), ds, sampler


@torch.no_grad()
def evaluate(model, loader, vocab, device, limit=None):
    model.eval()
    refs, hyps = [], []
    for n, (x, xl, y, yl, _) in enumerate(loader):
        logp, ol = model(x.to(device), xl.to(device))
        refs += decode_targets(y, yl, vocab)
        hyps += greedy_decode(logp.cpu(), ol.cpu(), vocab)
        if limit and n + 1 >= limit:
            break
    return wer(refs, hyps), cer(refs, hyps), refs[:3], hyps[:3]


def decode_targets(y, yl, vocab):
    out, off = [], 0
    for n in yl.tolist():
        out.append(vocab.decode(y[off : off + n].tolist()))
        off += n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="manifests/train.jsonl")
    p.add_argument("--valid", default="manifests/validation.jsonl")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--lstm-layers", type=int, default=0,
                   help="0 = pure CNN acoustic model; >0 adds a BiLSTM head")
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--max-dur", type=float, default=40.0,
                   help="drop longer train clips; corpus max is 35.9 s, so the "
                        "default keeps everything -- lower it only to cut memory")
    p.add_argument("--max-frames", type=int, default=1_600_000)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resume", default=None,
                   help="path to last.pt; restores model, optimizer, schedule "
                        "and epoch. Colab sessions die -- always pass this.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--eval-batches", type=int, default=30,
                   help="per-epoch validation is a SAMPLE this many batches "
                        "wide, to keep epochs cheap; a full-split eval runs "
                        "once at the end")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device. This model is not trainable on CPU in "
              "reasonable time -- run on a GPU (Kaggle/Colab).", flush=True)
    vocab = Vocab()
    tl, tds, tsamp = build_loader(args.train, vocab, args, True)
    vl, vds, _ = build_loader(args.valid, vocab, args, False)
    print(f"train {len(tds)} clips / {len(tsamp)} batches, valid {len(vds)} clips")

    model = ShonaCNN(args.n_mels, len(vocab), args.width,
                     lstm_layers=args.lstm_layers).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_par/1e6:.2f} M")

    ctc = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tsamp), pct_start=0.1)
    scaler = torch.amp.GradScaler(device, enabled=args.amp and device == "cuda")

    os.makedirs(args.out, exist_ok=True)
    best, start_ep = float("inf"), 1
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_ep = ck["epoch"] + 1
        best = ck.get("best", best)
        # OneCycleLR bakes total_steps into its state_dict, so restoring it
        # would crash the moment you resume with a larger --epochs. Rebuild the
        # schedule for the current run and fast-forward it instead.
        done = ck.get("gstep", (start_ep - 1) * len(tsamp))
        if done >= sched.total_steps:
            print(f"WARNING: {done} steps already done but this run's schedule "
                  f"is only {sched.total_steps} -- raise --epochs")
        for _ in range(min(done, sched.total_steps - 1)):
            sched.step()
        print(f"resumed from {args.resume} at epoch {start_ep} "
              f"(best CER {best:.4f}, lr {sched.get_last_lr()[0]:.2e})")
    elif args.resume:
        print(f"{args.resume} not found -- starting fresh")

    gstep = 0

    def save(path, ep, **extra):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args), "vocab": vocab.itos, "epoch": ep,
                    "best": best, "gstep": gstep, **extra}, path)

    gstep = (start_ep - 1) * len(tsamp)
    for ep in range(start_ep, args.epochs + 1):
        model.train(); tsamp.set_epoch(ep); run = 0.0; t0 = time.time()
        for i, (x, xl, y, yl, _) in enumerate(tl):
            x, xl, y, yl = x.to(device), xl.to(device), y.to(device), yl.to(device)
            with torch.amp.autocast(device, enabled=scaler.is_enabled()):
                logp, ol = model(x, xl)
                loss = ctc(logp.float(), y, ol, yl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            gstep += 1
            run += loss.item()
            if i % 50 == 0:
                print(f"ep{ep} {i}/{len(tsamp)} loss {run/(i+1):.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        w, c, refs, hyps = evaluate(model, vl, vocab, device, limit=args.eval_batches)
        print(f"== epoch {ep}: train loss {run/len(tsamp):.3f} "
              f"valid WER {w:.3f} CER {c:.3f} (SAMPLE of {args.eval_batches} "
              f"batches, not the full split) ({time.time()-t0:.0f}s)", flush=True)
        for r, h in zip(refs, hyps):
            print(f"   REF {r[:80]}\n   HYP {h[:80]}")
        if c < best:
            best = c
            save(os.path.join(args.out, "best.pt"), ep, cer=c)
        save(os.path.join(args.out, "last.pt"), ep)
    # Per-epoch numbers above are a sample; quote only this one.
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"),
                                     map_location=device)["model"])
    w, c, _, _ = evaluate(model, vl, vocab, device, limit=None)
    print(f"best-checkpoint FULL valid: WER {w:.4f} CER {c:.4f} "
          f"({len(vds)} clips)")


if __name__ == "__main__":
    main()
