"""Post-extraction prep: char vocab, LM text corpus, duration stats, leakage check."""
import json, os, collections, statistics as st

MAN = "manifests"
TEXT = "text"
SPLITS = ["train", "validation", "test"]

def load(split):
    p = os.path.join(MAN, f"{split}.jsonl")
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def main():
    os.makedirs(TEXT, exist_ok=True)
    data = {s: load(s) for s in SPLITS if os.path.exists(os.path.join(MAN, f"{s}.jsonl"))}

    # ---- duration stats -------------------------------------------------
    print(f"{'split':<11}{'clips':>7}{'hours':>8}{'min':>7}{'p50':>7}{'p95':>7}{'max':>7}{'spk':>6}")
    for s, rows in data.items():
        d = sorted(r["duration"] for r in rows)
        q = lambda p: d[min(len(d) - 1, int(p * len(d)))]
        print(f"{s:<11}{len(d):>7}{sum(d)/3600:>8.2f}{d[0]:>7.1f}{q(.5):>7.1f}"
              f"{q(.95):>7.1f}{d[-1]:>7.1f}{len({r['speaker_id'] for r in rows}):>6}")
    print(f"{'TOTAL':<11}{sum(len(v) for v in data.values()):>7}"
          f"{sum(r['duration'] for v in data.values() for r in v)/3600:>8.2f}")

    # ---- speaker leakage across splits ----------------------------------
    spk = {s: {r["speaker_id"] for r in rows} for s, rows in data.items()}
    print("\nspeaker overlap between splits:")
    for a in SPLITS:
        for b in SPLITS:
            if a < b and a in spk and b in spk:
                print(f"  {a} n {b}: {len(spk[a] & spk[b])}")

    # ---- char vocab (from train only) -----------------------------------
    train = data.get("train", [])
    chars = collections.Counter(c for r in train for c in r["text"])
    vocab = ["<blank>"] + [c for c, _ in sorted(chars.items())]
    with open(os.path.join(TEXT, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "blank_id": 0}, f, ensure_ascii=False, indent=1)
    print(f"\nvocab ({len(vocab)} symbols incl. blank): {' '.join(repr(c) for c in vocab[1:])}")
    print("char counts:", chars.most_common(8), "...rarest:", chars.most_common()[-5:])

    # ---- LM corpus (train transcripts) ----------------------------------
    with open(os.path.join(TEXT, "lm_train.txt"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(r["text"] + "\n")
    with open(os.path.join(TEXT, "lm_valid.txt"), "w", encoding="utf-8") as f:
        for r in data.get("validation", []):
            f.write(r["text"] + "\n")
    words = [w for r in train for w in r["text"].split()]
    print(f"\nLM corpus: {len(words)} tokens, {len(set(words))} unique words")
    print("top words:", [w for w, _ in collections.Counter(words).most_common(10)])

if __name__ == "__main__":
    main()
