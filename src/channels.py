"""Channel models applied to the transmitted symbols.

Symbols are tensors of shape (batch, n_symbols, 2) holding the real and
imaginary parts. Noise power is set from the batch's mean signal power and the
SNR in dB, so the SNR is the ratio of signal to noise power per symbol.
"""
import torch
import torch.nn as nn
import math
class AWGNChannel(nn.Module):
    """Additive white Gaussian noise: Y = X + N, N ~ CN(0, sigma^2 I)."""
    def __init__(self, snr_db=10.0):
        super().__init__(); self.snr_db = snr_db
    def forward(self, X):
        sig = X.pow(2).mean()
        std = (sig / (10.0**(self.snr_db/10.0) + 1e-12)).sqrt()
        return X + torch.randn_like(X) * std


class RayleighChannel(nn.Module):
    """Per-symbol Rayleigh fading with zero-forcing equalisation (perfect CSI).

    Y    = h*X + N,   h ~ CN(0, 1)
    Y_eq = Y / h

    The receiver divides by h, so the decoder sees equalised symbols and the
    AWGN and Rayleigh results are comparable at the same nominal SNR.
    """
    def __init__(self, snr_db=10.0):
        super().__init__(); self.snr_db = snr_db
    def forward(self, X):
        B, N, _ = X.shape; dev = X.device
        h_re = torch.randn(B,N,1,device=dev)/math.sqrt(2)
        h_im = torch.randn(B,N,1,device=dev)/math.sqrt(2)
        x_re, x_im = X[...,0:1], X[...,1:2]
        # Complex multiplication h*X in real arithmetic.
        y_re = h_re*x_re - h_im*x_im
        y_im = h_re*x_im + h_im*x_re
        sig = X.pow(2).mean()
        std = (sig/(10.0**(self.snr_db/10.0)+1e-12)).sqrt()
        y_re = y_re + torch.randn_like(y_re)*std
        y_im = y_im + torch.randn_like(y_im)*std
        # Y / h = Y * conj(h) / |h|^2; the small floor guards against h = 0.
        h_sq = h_re.pow(2) + h_im.pow(2) + 1e-12
        eq_re = (y_re*h_re + y_im*h_im)/h_sq
        eq_im = (y_im*h_re - y_re*h_im)/h_sq
        return torch.cat([eq_re, eq_im], dim=-1)


def _make_channel(ctype, snr_db):
    if ctype == "awgn":     return AWGNChannel(snr_db)
    if ctype == "rayleigh": return RayleighChannel(snr_db)
    raise ValueError(f"Unknown channel '{ctype}'")
