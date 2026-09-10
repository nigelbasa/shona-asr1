"""Decode Waxal Shona parquet shards -> 16 kHz mono FLAC + JSONL manifests.

Source MP3 is 128 kbps CBR mono/48 kHz, so byte length maps exactly to duration
(16000 B/s), but we take the true duration from the decoded stream anyway.
"""
import json, os, re, subprocess, sys
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
import pyarrow.parquet as pq

RAW = "data/raw/data/ASR/sna"
OUT_AUDIO = "data/audio"
OUT_MAN = "manifests"
SR = 16000
# Colab gives 2 vCPUs; this box has 8. ffmpeg-per-clip is the bottleneck.
WORKERS = int(os.environ.get("EXTRACT_WORKERS", os.cpu_count() or 4))

SPLITS = {
    "train": [f"sna-train-{i:05d}.parquet" for i in range(9)],
    "validation": [f"sna-validation-{i:05d}.parquet" for i in range(2)],
    "test": [f"sna-test-{i:05d}.parquet" for i in range(2)],
}

# Shona orthography: a-z plus the apostrophe in ng' (velar nasal). Everything
# else is punctuation we drop for CTC targets.
_KEEP = re.compile(r"[^a-z' ]+")

def normalize(t: str) -> str:
    t = t.lower().replace("’", "'").replace("ʼ", "'")
    t = _KEEP.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _decode(args):
    """mp3 bytes -> flac on disk. Returns (rel_path, n_samples) or None."""
    mp3, dest = args
    if os.path.exists(dest):
        try:
            return dest, sf.info(dest).frames
        except Exception:
            os.remove(dest)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR),
         "-sample_fmt", "s16", "-y", dest],
        input=mp3, capture_output=True)
    if p.returncode != 0 or not os.path.exists(dest):
        sys.stderr.write(f"FAIL {dest}: {p.stderr[:200]!r}\n")
        return None
    return dest, sf.info(dest).frames

def run_split(split, shards):
    adir = os.path.join(OUT_AUDIO, split)
    os.makedirs(adir, exist_ok=True)
    os.makedirs(OUT_MAN, exist_ok=True)
    man_path = os.path.join(OUT_MAN, f"{split}.jsonl")
    n = 0
    total_sec = 0.0
    with open(man_path, "w", encoding="utf-8") as man, \
         ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for shard in shards:
            path = os.path.join(RAW, shard)
            if not os.path.exists(path):
                sys.stderr.write(f"skip missing {shard}\n")
                continue
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                rows = pf.read_row_group(rg).to_pylist()
                jobs, meta = [], []
                for r in rows:
                    b = r["audio"]["bytes"]
                    uid = r["id"]
                    dest = os.path.join(adir, uid + ".flac")
                    jobs.append((b, dest))
                    meta.append((r, len(b)))
                for res, (r, nbytes) in zip(pool.map(_decode, jobs), meta):
                    if res is None:
                        continue
                    rel, frames = res
                    dur = frames / float(SR)   # true decoded duration
                    text = normalize(r["transcription"])
                    if not text:
                        continue
                    man.write(json.dumps({
                        "id": r["id"],
                        "audio": rel.replace("\\", "/"),
                        "duration": round(dur, 3),
                        "n_samples": frames,
                        "speaker_id": r["speaker_id"],
                        "gender": (r["gender"] or "").strip().lower(),
                        "text": text,
                        "text_raw": r["transcription"],
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    total_sec += dur
            print(f"  {shard}: running total {n} clips, {total_sec/3600:.2f} h", flush=True)
    print(f"{split}: {n} clips, {total_sec/3600:.2f} hours -> {man_path}", flush=True)
    return n, total_sec

if __name__ == "__main__":
    want = sys.argv[1:] or list(SPLITS)
    grand = 0.0
    for s in want:
        _, sec = run_split(s, SPLITS[s])
        grand += sec
    print(f"TOTAL {grand/3600:.2f} hours")
