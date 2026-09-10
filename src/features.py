"""log-mel front end with no torchaudio dependency (torch.stft + HTK mel bank)."""
import torch

def hz_to_mel(f):
    return 2595.0 * torch.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(n_mels=80, n_fft=400, sr=16000, f_min=20.0, f_max=7600.0):
    """(n_mels, n_fft//2+1) triangular filters, slaney-free HTK scale."""
    n_freq = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_freq)
    pts = mel_to_hz(torch.linspace(hz_to_mel(torch.tensor(f_min)),
                                   hz_to_mel(torch.tensor(f_max)), n_mels + 2))
    fb = torch.zeros(n_mels, n_freq)
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        left = (freqs - lo) / torch.clamp(mid - lo, min=1e-6)
        right = (hi - freqs) / torch.clamp(hi - mid, min=1e-6)
        fb[i] = torch.clamp(torch.minimum(left, right), min=0.0)
    return fb


class LogMel(torch.nn.Module):
    def __init__(self, sr=16000, n_fft=400, hop=160, n_mels=80,
                 f_min=20.0, f_max=7600.0):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("fb", mel_filterbank(n_mels, n_fft, sr, f_min, f_max))

    def forward(self, wav):
        """wav: (T,) float -> (n_mels, frames), per-utterance mean/var normalized."""
        spec = torch.stft(wav, self.n_fft, self.hop, self.n_fft, self.window,
                          center=True, pad_mode="reflect", return_complex=True)
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = (self.fb @ power + 1e-6).log()
        return (mel - mel.mean(1, keepdim=True)) / (mel.std(1, keepdim=True) + 1e-5)


def spec_augment(feat, n_freq_masks=2, freq_width=27, n_time_masks=2,
                 time_frac=0.05, generator=None):
    """In-place-safe SpecAugment on a (n_mels, T) tensor."""
    feat = feat.clone()
    n_mels, T = feat.shape
    for _ in range(n_freq_masks):
        w = int(torch.randint(0, freq_width + 1, (1,), generator=generator))
        if w and n_mels > w:
            f0 = int(torch.randint(0, n_mels - w, (1,), generator=generator))
            feat[f0 : f0 + w, :] = 0.0
    max_t = max(1, int(T * time_frac))
    for _ in range(n_time_masks):
        w = int(torch.randint(0, max_t + 1, (1,), generator=generator))
        if w and T > w:
            t0 = int(torch.randint(0, T - w, (1,), generator=generator))
            feat[:, t0 : t0 + w] = 0.0
    return feat
