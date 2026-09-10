"""Decode a manifest with a trained checkpoint: greedy, and optionally
prefix beam search with LSTM-LM shallow fusion. Prints WER/CER for each.

Beam search here is genuinely slow: _lm_step re-runs the LSTM over the whole
prefix on every cache miss, so cost grows as roughly beam x frames x prefix.
Start with --limit 20 --beam 8 to measure throughput before decoding a whole
split.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import DataLoader

from data import ShonaASR, Vocab, collate, FrameBudgetSampler
from model import ShonaCNN
from lm import CharLM
from decode import greedy_decode, beam_decode, wer, cer


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--manifest", default="manifests/test.jsonl")
    p.add_argument("--lm", default=None,
                   help="LM checkpoint. Must match the protocol: lm_spk.pt for "
                        "spk_* manifests, lm_official.pt for the official ones.")
    p.add_argument("--beam", type=int, default=0, help="0 = greedy only")
    p.add_argument("--lm-weight", type=float, default=0.4)
    p.add_argument("--limit", type=int, default=None, help="first N clips only")
    p.add_argument("--max-batch", type=int, default=16)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    vocab = Vocab()
    ma = ck.get("args", {})
    model = ShonaCNN(ma.get("n_mels", 80), len(vocab), ma.get("width", 256),
                     lstm_layers=ma.get("lstm_layers", 0)).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"acoustic: {args.ckpt} (epoch {ck.get('epoch')})")

    lm = None
    if args.lm:
        lc = torch.load(args.lm, map_location=device, weights_only=False)
        lm = CharLM(len(vocab)).to(device); lm.load_state_dict(lc["model"]); lm.eval()
        print(f"LM: {args.lm} (trained on {lc.get('train_txt')}, "
              f"ppl {lc.get('valid_ppl')})")
        if ("spk" in os.path.basename(args.manifest)) != ("spk" in str(lc.get("train_txt"))):
            print("  !! LM corpus and manifest are from different protocols -- "
                  "results will be contaminated", file=sys.stderr)

    ds = ShonaASR(args.manifest, vocab, n_mels=ma.get("n_mels", 80), train=False)
    if args.limit:
        ds.rows = ds.rows[: args.limit]
    dl = DataLoader(ds, batch_sampler=FrameBudgetSampler(ds, max_batch=args.max_batch,
                                                         shuffle=False),
                    collate_fn=collate, num_workers=0)

    refs, gre, bea = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for x, xl, y, yl, _ in dl:
            logp, ol = model(x.to(device), xl.to(device))
            logp, ol = logp.cpu(), ol.cpu()
            off = 0
            for n in yl.tolist():
                refs.append(vocab.decode(y[off : off + n].tolist())); off += n
            gre += greedy_decode(logp, ol, vocab)
            if args.beam:
                for b in range(logp.shape[1]):
                    bea.append(beam_decode(logp[: ol[b], b], vocab, beam=args.beam,
                                           lm=lm, lm_weight=args.lm_weight))
            print(f"  {len(refs)}/{len(ds)} clips ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{len(refs)} clips, {sum(r['duration'] for r in ds.rows)/3600:.2f} h")
    print(f"{'greedy':<14}WER {wer(refs, gre):.4f}  CER {cer(refs, gre):.4f}")
    if args.beam:
        tag = f"beam{args.beam}" + (f"+LM({args.lm_weight})" if lm else "")
        print(f"{tag:<14}WER {wer(refs, bea):.4f}  CER {cer(refs, bea):.4f}")
    for r, h in list(zip(refs, bea or gre))[:3]:
        print(f"  REF {r[:90]}\n  HYP {h[:90]}")


if __name__ == "__main__":
    main()
