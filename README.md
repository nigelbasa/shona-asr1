# shona-asr1

Shona (`sna`) ASR on the **Waxal** corpus — [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP),
config `sna_asr`. CNN acoustic model with CTC, plus an LSTM character LM for
shallow fusion at decode time.

## What's here

**99.4 hours** of labeled, transcribed Shona speech — the complete labeled
portion of the Waxal Shona set (the 26 GB `unlabeled` split was not pulled).

| split | clips | hours | speakers |
|---|---|---|---|
| train | 14,109 | 79.73 | 157 |
| validation | 1,727 | 9.71 | 100 |
| test | 1,749 | 9.94 | 115 |
| **total** | **17,585** | **99.38** | 168 |

Audio: 16 kHz mono FLAC in `data/audio/<split>/<id>.flac` (5.0 GB), decoded from
the source 48 kHz 128 kbps MP3. Manifests are JSONL in `manifests/`, one object
per clip: `id, audio, duration, n_samples, speaker_id, gender, text, text_raw`.

## Two things to know about this data

**1. The official splits share speakers.** 97 of 100 validation speakers and 106
of 115 test speakers also appear in train. WER measured on the official splits is
speaker-dependent and will flatter the model. `scripts/speaker_split.py`
repartitions the **official train split** so no speaker crosses a boundary:

| split | clips | hours | speakers |
|---|---|---|---|
| spk_train | 12,464 | 70.31 | 117 |
| spk_dev | 903 | 5.31 | 20 |
| spk_test | 742 | 4.11 | 20 |

These are drawn from official train *only* (the three partition it exactly:
12,464 + 903 + 742 = 14,109). That matters: pooling all three official splits
would put official-test clips into `spk_train`, and then no checkpoint could
honestly be scored on the official test set at all.

Run these as **two separate protocols**, and never mix a checkpoint across them:

- **Protocol A — paper-comparable.** Train `train.jsonl`, validate
  `validation.jsonl`, test `test.jsonl`. LM: `text/lm_train.txt`.
  Carries the speaker-overlap caveat above.
- **Protocol B — honest generalization.** Train `spk_train.jsonl`, validate
  `spk_dev.jsonl`, test `spk_test.jsonl`. LM: `text/lm_spk_train.txt`.
  Every test voice is unseen.

A Protocol B checkpoint is also clip-clean against official validation/test, so
it can additionally be reported there — but that number still inherits the
speaker overlap, so label it as such.

**Do not decode Protocol B with `text/lm_train.txt`.** That corpus contains the
transcript of every `spk_test` clip (742 of 742), so shallow fusion would be
scoring against sentences the LM has memorized. `speaker_split.py` writes
`text/lm_spk_train.txt` and `lm_spk_dev.txt` for exactly this reason.

**2. Clips are long and speaker durations are very skewed.** Median clip is
19.7 s (p95 29.1 s, max 35.9 s) — several times the length CTC recipes usually
assume, so batching is by *frame budget* with a batch cap, not fixed batch size
(`FrameBudgetSampler`). Per-speaker totals range from 9.8 h down to seconds,
median 3.4 min; a handful of voices dominate the corpus, which is why the
held-out splits cap how much any one speaker can contribute.

## Text

29-symbol character vocabulary (`text/vocab.json`): `<blank>`, space, apostrophe,
`a`–`z`. The apostrophe is kept because Shona `ng'` is a distinct velar nasal.
Normalization lowercases, strips punctuation, and collapses whitespace — it drops
nothing else (0 clips lost). `l`, `x`, `q` occur only in loanwords and are rare
(658 / 22 / 9 occurrences).

Character CTC handles the Shona digraphs (`sv`, `zv`, `dz`, `mh`, `ng'`) without
a special tokenizer.

LM corpora: `text/lm_train.txt` (332 k word tokens, 36,889 unique words) for
Protocol A, `text/lm_spk_train.txt` for Protocol B.

## Layout

```
scripts/download.py        pull sna parquet shards from HF
scripts/extract.py         parquet -> 16 kHz FLAC + JSONL manifests
scripts/prepare.py         vocab, LM corpus, duration stats, leakage check
scripts/speaker_split.py   speaker-disjoint repartition
src/features.py            log-mel front end (torch.stft + HTK mel bank)
src/data.py                Dataset, Vocab, frame-budget batching
src/model.py               ShonaCNN: QuartzNet-style separable-conv CTC model
src/lm.py                  CharLM: LSTM character LM
src/decode.py              greedy + prefix beam search w/ LM fusion, WER/CER
src/train.py               training loop
src/eval.py                decode a manifest: greedy and beam+LM fusion
notebooks/                 Colab notebook (scripts/make_colab_nb.py builds it)
```

`torchaudio` is deliberately not used — its latest release pins an older torch
and `torchaudio.load` is on the way out. The front end is ~40 lines of
`torch.stft` instead.

## Model

`ShonaCNN` is a QuartzNet-style **1D CNN**: two stride-2 separable convs
(4× time subsample, 40 ms frames), five residual blocks of depthwise-separable
convs with growing kernels (13→25), a dilated head conv, then a 1×1 CTC
projection. ~3.4 M params at the default `width=256`. It is a *pure CNN* by
default; `--lstm-layers 2` adds a BiLSTM head if you want the hybrid.

The LSTM in `src/lm.py` is separate — a character LM trained on transcripts
only, used for shallow fusion in `beam_decode`, not part of the acoustic path.

## Running

One command per line - Windows PowerShell 5.1 has no `&&`:

```
pip install -r requirements.txt
python scripts/download.py
python scripts/extract.py
python scripts/prepare.py
python scripts/speaker_split.py
```

Protocol A:

```bash
python src/train.py --train manifests/train.jsonl --valid manifests/validation.jsonl --amp
```

Protocol B:

```bash
python src/train.py --train manifests/spk_train.jsonl --valid manifests/spk_dev.jsonl --amp
```

Per-epoch WER/CER is a *sample* of `--eval-batches` batches to keep epochs
cheap; a full-split evaluation of the best checkpoint runs once at the end. Quote
that one.

Character LM — the corpus determines the checkpoint name, so the two protocols'
LMs cannot overwrite each other:

```bash
python src/lm.py --train text/lm_train.txt      # -> checkpoints/lm_official.pt
python src/lm.py --train text/lm_spk_train.txt  # -> checkpoints/lm_spk.pt
```

Decoding. `eval.py` warns if the LM corpus and the manifest come from different
protocols:

```bash
python src/eval.py --manifest manifests/spk_test.jsonl --lm checkpoints/lm_spk.pt --beam 8
```

Beam search with fusion is slow — `_lm_step` re-runs the LSTM over the whole
prefix on each cache miss, so cost grows roughly as beam x frames x prefix
length. Measure with `--limit 20 --beam 8` before turning it loose on a full
split; `--beam 0` (the default) is greedy only and fast. Threading the LSTM
state incrementally is the obvious optimization and is not done yet.

### Colab

`notebooks/shona_asr_colab.ipynb` is the working notebook: GPU check, Drive
mount, corpus build, protocol switch, throughput probe, train, LM, decode.

The one idea worth knowing up front: **build the corpus once on Colab and tar it
to Drive.** Decoding 17,585 clips through ffmpeg on 2 vCPUs takes 1-2 hours, and
you do not want to pay that every session. After the first run the notebook
untars from Drive in minutes. Do not try to upload the 5 GB from here.

`train.py --resume` restores model, optimizer, LR schedule and epoch, so a
dropped Colab session costs one epoch at most. Point `--out` and `--resume` at a
Drive path and re-run the same cell to continue.

### You need a GPU

This machine has an Intel UHD 620 and no CUDA device. 80 h of 20-second audio is
not trainable here in any useful time — `train.py` warns and continues, which is
fine for a shape/loss check on a few batches, not for real training. Run on
Kaggle or Colab; the code is device-agnostic and `--amp` enables mixed precision
on CUDA.

Because the corpus is ~5 GB, pull it directly from HF inside the notebook with
`scripts/download.py` + `scripts/extract.py` rather than uploading it.

## Source

Waxal is CC-BY-SA-4.0 / CC-BY-4.0. Paper: *WAXAL: A Large-Scale Multilingual
African Language Speech Corpus* ([arXiv:2602.02734](https://arxiv.org/abs/2602.02734)).

`data/raw/` (the 5.2 GB of source parquet) has been deleted — it is only needed
to re-extract, and `scripts/download.py` fetches it again on demand.
