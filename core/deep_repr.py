"""OPTIONAL: a *properly specified* 1-D convolutional autoencoder embedding.

Included to show the contrast with the thesis's dense AE. Key differences that
matter:
  * 1-D convolutions with weight sharing exploit the temporal structure the dense
    AE ignored, with ~4-5 orders of magnitude fewer parameters than 10000-wide
    dense layers.
  * Trained to convergence with a validation split, not stopped at epoch 1.
  * A sensible bottleneck (32-64), not 6.

Even so, a *reconstruction* AE optimises for energy, not class structure, so this
is expected to trail the rhythm/morph features for arrhythmia discovery. The
recommended deep alternative for a follow-up is self-supervised CONTRASTIVE
pretraining (e.g. augmentation-invariant embeddings), which targets
discriminability directly. Left as documented future work.
"""
from __future__ import annotations

import numpy as np


def _resample_stack(recordings, target_len=2500):
    """Resample every recording to (n_leads, target_len) for a fixed-size tensor."""
    from scipy.signal import resample
    X = []
    for r in recordings:
        sig = r.signal.T                                   # (n_leads, n_samples)
        X.append(resample(sig, target_len, axis=1))
    return np.asarray(X, dtype=np.float32)                 # (N, leads, target_len)


def deep_ae_embedding(recordings, latent=32, epochs=50, batch=64, lr=1e-3, seed=0):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    torch.manual_seed(seed)
    X = _resample_stack(recordings)
    # per-lead z-score so amplitude scale doesn't dominate
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)
    C, T = X.shape[1], X.shape[2]
    t = torch.tensor(X)

    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(C, 32, 7, stride=2, padding=3), nn.LeakyReLU(),
                nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.LeakyReLU(),
                nn.Conv1d(64, 128, 7, stride=2, padding=3), nn.LeakyReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, latent))
            self.dec_fc = nn.Linear(latent, 128 * (T // 8))
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(128, 64, 8, stride=2, padding=3), nn.LeakyReLU(),
                nn.ConvTranspose1d(64, 32, 8, stride=2, padding=3), nn.LeakyReLU(),
                nn.ConvTranspose1d(32, C, 8, stride=2, padding=3))

        def forward(self, x):
            import torch.nn.functional as F
            z = self.enc(x)
            h = self.dec_fc(z).view(x.size(0), 128, T // 8)
            out = self.dec(h)
            # transpose-conv length rarely lands exactly on T -> align (crop or pad)
            if out.size(-1) > x.size(-1):
                out = out[..., :x.size(-1)]
            elif out.size(-1) < x.size(-1):
                out = F.pad(out, (0, x.size(-1) - out.size(-1)))
            return out, z

    model = ConvAE()
    ds = TensorDataset(t)
    n_val = max(1, int(0.1 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for (xb,) in tl:
            opt.zero_grad()
            out, _ = model(xb)
            loss = loss_fn(out, xb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vx = va[:][0]
            vloss = loss_fn(model(vx)[0], vx).item()
        if vloss < best - 1e-5:
            best, best_state, patience = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, z = model(t)
    return z.numpy()
