"""Download Waxal Shona (sna_asr) labeled parquet shards from google/WaxalNLP."""
import sys, time
from huggingface_hub import hf_hub_download

REPO = "google/WaxalNLP"
BASE = "data/ASR/sna"
LOCAL = "data/raw"

def files(train_shards):
    fs = [f"{BASE}/sna-validation-{i:05d}.parquet" for i in range(2)]
    fs += [f"{BASE}/sna-test-{i:05d}.parquet" for i in range(2)]
    fs += [f"{BASE}/sna-train-{i:05d}.parquet" for i in train_shards]
    return fs

if __name__ == "__main__":
    shards = [int(x) for x in sys.argv[1:]] or [0, 1, 2]
    for f in files(shards):
        t = time.time()
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=LOCAL)
        print(f"ok {f}  {time.time()-t:.0f}s", flush=True)
    print("DONE")
