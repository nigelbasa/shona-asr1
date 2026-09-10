"""CTC decoding (greedy + beam search with LSTM-LM shallow fusion) and metrics."""
import math
import torch


def greedy_decode(logp, out_lengths, vocab):
    """logp: (T, B, C) -> list of strings (collapse repeats, drop blanks)."""
    ids = logp.argmax(-1).transpose(0, 1)          # (B, T)
    hyps = []
    for row, n in zip(ids.tolist(), out_lengths.tolist()):
        prev, out = -1, []
        for k in row[:n]:
            if k != prev and k != vocab.blank:
                out.append(k)
            prev = k
        hyps.append("".join(vocab.itos[i] for i in out))
    return hyps


def beam_decode(logp_seq, vocab, beam=16, lm=None, lm_weight=0.4,
                insert_bonus=None, device="cpu"):
    """Prefix beam search over one utterance. logp_seq: (T, C) log-probs.

    lm is a CharLM; shallow fusion adds lm_weight * log P_lm(c | prefix).
    """
    # An insertion bonus only makes sense to offset an LM's per-character
    # penalty; applied without one it just biases toward over-insertion.
    if insert_bonus is None:
        insert_bonus = 0.6 if lm is not None else 0.0
    # Prefix->logprob memo is only ever reused within this utterance.
    cache = {}
    NEG = -float("inf")
    beams = {(): (0.0, NEG, None)}                 # prefix -> (p_blank, p_nonblank, lm_state)
    T, C = logp_seq.shape
    for t in range(T):
        step = logp_seq[t]
        nxt = {}
        # prune the symbol set: full vocab x beam is wasteful at char level
        top = torch.topk(step, min(C, beam)).indices.tolist()
        if vocab.blank not in top:
            top.append(vocab.blank)
        for prefix, (pb, pnb, state) in beams.items():
            ptot = _logsum(pb, pnb)
            for c in top:
                p = step[c].item()
                if c == vocab.blank:
                    e = nxt.get(prefix, (NEG, NEG, state))
                    nxt[prefix] = (_logsum(e[0], ptot + p), e[1], state)
                    continue
                last = prefix[-1] if prefix else None
                if c == last:
                    # repeat without blank collapses onto the same prefix
                    e = nxt.get(prefix, (NEG, NEG, state))
                    nxt[prefix] = (e[0], _logsum(e[1], pnb + p), state)
                    src = pb                     # extending needs a blank between
                else:
                    src = ptot
                new = prefix + (c,)
                bonus = insert_bonus
                if lm is not None:
                    lp, st = _lm_step(lm, prefix, c, state, vocab, device, cache)
                    bonus += lm_weight * lp
                else:
                    st = state
                e = nxt.get(new, (NEG, NEG, st))
                nxt[new] = (e[0], _logsum(e[1], src + p + bonus), st)
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -_logsum(kv[1][0], kv[1][1]))[:beam])
    best = max(beams.items(), key=lambda kv: _logsum(kv[1][0], kv[1][1]))[0]
    return "".join(vocab.itos[i] for i in best)


def _lm_step(lm, prefix, c, state, vocab, device, cache):
    if prefix not in cache:
        inp = torch.tensor([[vocab.blank] + list(prefix)], device=device)
        with torch.no_grad():
            logits, _ = lm(inp)
        cache[prefix] = logits[0, -1].log_softmax(-1).cpu()
    return cache[prefix][c].item(), state


def _logsum(a, b):
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _levenshtein(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(refs, hyps):
    e = n = 0
    for r, h in zip(refs, hyps):
        rw = r.split()
        e += _levenshtein(rw, h.split()); n += len(rw)
    return e / max(n, 1)


def cer(refs, hyps):
    e = n = 0
    for r, h in zip(refs, hyps):
        e += _levenshtein(list(r), list(h)); n += len(r)
    return e / max(n, 1)
