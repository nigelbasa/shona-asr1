"""Generate notebooks/shona_asr_colab.ipynb."""
import json, os

def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}

def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}

BS = "\\"   # line-continuation backslash inside the generated shell cells

cells = [
md("""# Shona ASR on Colab - CNN-CTC on Waxal `sna`

Run the cells in order. **Runtime -> Change runtime type -> T4 GPU** first.

The pattern: build the corpus once, park it on Drive as a tarball, and every
later session just untars it. Extraction is the slow part and you only want to
pay for it once.
"""),

md("## 1. Confirm you actually got a GPU"),
code("""!nvidia-smi
import torch
print(torch.__version__, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")"""),
md("If `cuda.is_available()` is False, stop and fix the runtime type. "
   "Everything below assumes a GPU."),

md("## 2. Mount Drive\n\nCheckpoints and the corpus tarball live here, so a "
   "dropped session costs nothing."),
code("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE = '/content/drive/MyDrive/shona-asr1'
os.makedirs(DRIVE + '/checkpoints', exist_ok=True)
print(DRIVE)"""),

md("""## 3. Get the code

Push the project from your laptop once:

```bash
cd /c/Users/User/mydev/shona-asr1
git init && git add -A && git commit -m "Waxal sna pipeline + CNN-CTC model"
gh repo create shona-asr1 --private --source=. --push
```

`.gitignore` already excludes `data/` and `checkpoints/`, so this pushes code and
manifests (a few MB) - not the 5 GB of audio.

Then clone it here."""),
code("""REPO = 'YOUR_GITHUB_USERNAME/shona-asr1'   # <- edit this

%cd /content
![ -d shona-asr1 ] || git clone https://github.com/$REPO.git
%cd /content/shona-asr1
!git pull -q
!ls"""),
md("No GitHub? Zip the folder locally (excluding `data/`), drag it into Colab's "
   "file pane, then `!unzip -q shona-asr1.zip`. Re-uploading every session gets "
   "old fast, which is why the repo is worth five minutes."),

md("## 4. Dependencies\n\nTorch ships with Colab. We need the audio/data bits "
   "only - and no `torchaudio`, deliberately."),
code("""!pip install -q soundfile pyarrow "huggingface_hub>=1.0"
!apt-get -qq install -y ffmpeg > /dev/null
!ffmpeg -version | head -1"""),

md("""## 5. The corpus

First run: download the parquet from HF and decode to FLAC, then tar it to Drive.
On Colab's connection the download is a couple of minutes; **the decode is the
slow part** - 17,585 clips through ffmpeg on 2 vCPUs, so budget 1-2 hours.

Every later session takes the fast path and untars from Drive in minutes."""),
code("""import os, time
TARBALL = DRIVE + '/shona_audio_16k.tar'

if os.path.exists(TARBALL):
    print('restoring from Drive...')
    t = time.time()
    !tar -xf "$TARBALL" -C /content/shona-asr1
    print(f'restored in {time.time()-t:.0f}s')
else:
    print('building corpus from scratch (one time only)')
    !python scripts/download.py
    !python scripts/extract.py
    !python scripts/prepare.py
    !python scripts/speaker_split.py
    print('archiving to Drive so this never runs again...')
    !tar -cf "$TARBALL" data/audio manifests text
    !rm -rf data/raw          # 5.2 GB of source parquet, no longer needed"""),
code("""# Sanity check: every manifest row has its audio file
import json, glob, os
for sp in ('train', 'validation', 'test'):
    rows = [json.loads(l) for l in open(f'manifests/{sp}.jsonl', encoding='utf-8')]
    n = len(glob.glob(f'data/audio/{sp}/*.flac'))
    h = sum(r['duration'] for r in rows) / 3600
    miss = sum(1 for r in rows if not os.path.exists(r['audio']))
    print(f'{sp:<11}{len(rows):>6} clips  {h:>6.2f} h  {n} flac  missing {miss}')"""),

md("""## 6. Pick a protocol

Read this before training - it decides which numbers you are allowed to report.

- **Protocol A (paper-comparable):** `train` / `validation` / `test`. The
  official splits, but they share speakers (97 of 100 validation speakers are
  also in train), so the WER is speaker-dependent and flattering.
- **Protocol B (honest generalization):** `spk_train` / `spk_dev` / `spk_test`.
  Speaker-disjoint - every test voice is unseen.

**Never mix a checkpoint across protocols, and match the LM to the protocol.**
`text/lm_train.txt` contains the transcript of all 742 `spk_test` clips, so
decoding Protocol B with it would score against sentences the LM memorised."""),
code("""PROTOCOL = 'B'    # 'A' or 'B'

if PROTOCOL == 'B':
    TRAIN, VALID, TEST = 'spk_train', 'spk_dev', 'spk_test'
    LM_TRAIN, LM_VALID, LM_NAME = 'text/lm_spk_train.txt', 'text/lm_spk_dev.txt', 'lm_spk.pt'
else:
    TRAIN, VALID, TEST = 'train', 'validation', 'test'
    LM_TRAIN, LM_VALID, LM_NAME = 'text/lm_train.txt', 'text/lm_valid.txt', 'lm_official.pt'
print(PROTOCOL, TRAIN, VALID, TEST, LM_TRAIN, LM_NAME)"""),

md("""## 7. Measure throughput before committing

Do not launch 60 epochs on a guess. Run one short epoch on a slice and see what
a step actually costs, then work out how many epochs fit in a session."""),
code("""!head -600 manifests/$TRAIN.jsonl > manifests/_probe.jsonl
!python src/train.py --train manifests/_probe.jsonl --valid manifests/$VALID.jsonl """ + BS + """
    --epochs 1 --amp --eval-batches 3 --out /content/probe
!rm -rf /content/probe manifests/_probe.jsonl"""),
md("Scale the epoch time by `len(train)/600` for a real epoch. If that is over "
   "~40 minutes, drop `--width` to 192 or raise `--max-batch` before the real run."),

md("""## 8. Train

Checkpoints go straight to Drive and `--resume` points at the same file, so this
cell is also the *restart* cell. When Colab drops you, re-run cells 2-6 and then
this one - it picks up at the next epoch with the right learning rate."""),
code("""CKPT = DRIVE + '/checkpoints'

!python src/train.py """ + BS + """
    --train manifests/$TRAIN.jsonl """ + BS + """
    --valid manifests/$VALID.jsonl """ + BS + """
    --epochs 60 --lr 3e-4 --width 256 --amp """ + BS + """
    --max-frames 1600000 --max-batch 32 --workers 2 """ + BS + """
    --out "$CKPT" --resume "$CKPT/last.pt\""""),
md("Per-epoch WER/CER during training is a **sample** of `--eval-batches` "
   "batches, for the curve only. The full-split number prints once at the end.\n\n"
   "`--lstm-layers 2` swaps the pure CNN for a CNN+BiLSTM hybrid if you want to "
   "compare - but that is a different model, so give it its own `--out`."),

md("## 9. Character LM\n\nSeparate from the acoustic model. Minutes, not hours."),
code("""!python src/lm.py --train $LM_TRAIN --valid $LM_VALID """ + BS + """
    --epochs 10 --out "$CKPT/$LM_NAME\""""),

md("""## 10. Decode the test set

Greedy first - it is fast and it is your baseline. Beam + LM fusion is slow
(`_lm_step` re-runs the LSTM over the whole prefix on each cache miss), so
measure it on `--limit 20` before turning it loose on the full split."""),
code("""# greedy, full test set
!python src/eval.py --ckpt "$CKPT/best.pt" --manifest manifests/$TEST.jsonl"""),
code("""# beam + LM fusion, small slice first to see what it costs
!python src/eval.py --ckpt "$CKPT/best.pt" --manifest manifests/$TEST.jsonl """ + BS + """
    --lm "$CKPT/$LM_NAME" --beam 8 --lm-weight 0.4 --limit 20"""),

md("""## Notes that will save you a session

- **Free Colab disconnects on idle** (~90 min) and caps sessions near 12 h.
  Checkpointing to Drive every epoch is what makes that survivable - do not
  train to a local `--out`.
- **Drive writes can be slow.** If per-epoch checkpointing drags, write to
  `/content/checkpoints` and copy to Drive every few epochs instead.
- **Clips are long** (median 19.7 s, max 35.9 s). Batching is by frame budget,
  not clip count - if you hit OOM, lower `--max-frames`, not `--max-batch`.
- **A first real run will look terrible.** CTC from scratch spends early epochs
  emitting blanks and repeated characters; CER only moves once it finds the
  alignment. Judge it at epoch 10, not epoch 2.
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

os.makedirs("notebooks", exist_ok=True)
out = "notebooks/shona_asr_colab.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {out}: {len(cells)} cells")
