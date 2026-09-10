"""Build a speaker-disjoint train/dev/test split from the OFFICIAL TRAIN SPLIT.

The official Waxal sna splits share speakers (97/100 validation speakers and
106/115 test speakers also appear in train), so WER on them is speaker-dependent
and optimistic. This repartitions so no speaker crosses a boundary.

It draws ONLY from official train. Pooling all three splits would put official
test clips into spk_train, and then a spk_train model could not be scored on the
official test set at all. Sourcing from train alone keeps a single checkpoint
clip-clean against official validation/test as well.
"""
import json, os, random, collections

MAN = "manifests"
TARGET = {"spk_train": 0.90, "spk_dev": 0.05, "spk_test": 0.05}

def main(seed=13, n_giants=3, min_speakers=20):
    rows = [json.loads(l) for l in
            open(os.path.join(MAN, "train.jsonl"), encoding="utf-8")]

    by_spk = collections.defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    total = sum(r["duration"] for r in rows)

    # Durations are extremely skewed (top speaker 9.8 h, median 3.4 min), so a
    # naive greedy fill puts one giant voice in dev and calls it a split. Keep
    # the few giants in train and hold out many mid-sized speakers instead.
    order = sorted(by_spk, key=lambda s: -sum(r["duration"] for r in by_spk[s]))
    dur = {s: sum(r["duration"] for r in by_spk[s]) for s in order}
    assign = {s: "spk_train" for s in order[:n_giants]}

    quota = {"spk_dev": TARGET["spk_dev"] * total,
             "spk_test": TARGET["spk_test"] * total}
    got = {"spk_dev": 0.0, "spk_test": 0.0}
    cnt = {"spk_dev": 0, "spk_test": 0}
    rest = order[n_giants:]
    random.Random(seed).shuffle(rest)
    for spk in rest:
        need = [k for k in quota
                if got[k] < quota[k] or cnt[k] < min_speakers]
        if not need:
            assign[spk] = "spk_train"
            continue
        k = min(need, key=lambda k: got[k] / quota[k])
        # Cap how much any one voice can contribute. Once the duration quota
        # is met we are only still filling the speaker count, so accept just
        # small speakers -- otherwise a single big voice blows past the quota.
        cap = 0.5 * quota[k] if got[k] < quota[k] else 0.05 * quota[k]
        if dur[spk] > cap:
            assign[spk] = "spk_train"      # too big to hold out cleanly
            continue
        assign[spk] = k
        got[k] += dur[spk]
        cnt[k] += 1
    for spk in order:
        assign.setdefault(spk, "spk_train")

    print(f"{'split':<11}{'clips':>7}{'hours':>8}{'share':>8}{'spk':>6}")
    for k in TARGET:
        sel = [r for spk, kk in assign.items() if kk == k for r in by_spk[spk]]
        sel.sort(key=lambda r: r["id"])
        with open(os.path.join(MAN, f"{k}.jsonl"), "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = sum(r["duration"] for r in sel) / 3600
        print(f"{k:<11}{len(sel):>7}{h:>8.2f}{h*3600/total:>8.1%}"
              f"{len({r['speaker_id'] for r in sel}):>6}")
        if k in ("spk_train", "spk_dev"):
            # A LM trained on text/lm_train.txt has memorised spk_dev/spk_test
            # sentences, which would flatter shallow fusion. Emit clean corpora.
            tag = "lm_spk_train" if k == "spk_train" else "lm_spk_dev"
            with open(os.path.join("text", f"{tag}.txt"), "w", encoding="utf-8") as f:
                for r in sel:
                    f.write(r["text"] + chr(10))

    sets = {k: {spk for spk, kk in assign.items() if kk == k} for k in TARGET}
    ks = list(TARGET)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            n = len(sets[ks[i]] & sets[ks[j]])
            print(f"  overlap {ks[i]} n {ks[j]}: {n}")
            assert n == 0

if __name__ == "__main__":
    main()
