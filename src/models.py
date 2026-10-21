"""Deep JSCC system and the two anomaly detectors.

    x -> SwinEncoder -> z -> JsccMapper -> X -> channel -> Y -> JsccDecoder -> z'
      -> BottleneckDecoder -> x_hat

z and z' are flat vectors of patch_grid * patch_grid * patch_dim values (1,568 for
both production systems). The detectors score the received latent z', viewed as a
patch_grid x patch_grid grid of patch_dim-dimensional cells.
"""
import torch
import torch.nn as nn
import timm
from src.channels import _make_channel
class SwinEncoder(nn.Module):
    """Swin-Tiny (timm, ImageNet-pretrained) feature map of one stage, pooled to a
    patch_grid x patch_grid grid and reduced to patch_dim channels by a 1x1 conv.

    Stage channels and native grids: 0 -> 96 at 56x56, 1 -> 192 at 28x28,
    2 -> 384 at 14x14, 3 -> 768 at 7x7.
    """
    def __init__(self, pretrained=True, patch_grid=8, patch_dim=16, stage_idx=2):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=pretrained,
            features_only=True, out_indices=(0,1,2,3))
        self.patch_grid = patch_grid
        self.patch_dim  = patch_dim
        self.stage_idx  = stage_idx
        _stage_channels = {0: 96, 1: 192, 2: 384, 3: 768}[stage_idx]
        self.spatial_pool   = nn.AdaptiveAvgPool2d((patch_grid, patch_grid))
        self.channel_reduce = nn.Conv2d(_stage_channels, patch_dim, kernel_size=1)
        self.latent_dim = patch_grid * patch_grid * patch_dim

    def forward(self, x):
        feats = self.backbone(x)
        skips = [f.permute(0,3,1,2).contiguous() for f in feats]  # BHWC -> BCHW
        chosen  = skips[self.stage_idx]             # [B, C_stage, H_stage, W_stage]
        pooled  = self.spatial_pool(chosen)          # [B, C_stage, P, P]
        reduced = self.channel_reduce(pooled)        # [B, patch_dim, P, P]
        z = reduced.flatten(1)                       # [B, patch_dim*P*P]
        return z, reduced


class JsccMapper(nn.Module):
    """Latent z -> n_symbols complex symbols (shape [B, n_symbols, 2]) with unit
    average power per image."""
    def __init__(self, latent_dim=768, n_symbols=2048):
        super().__init__()
        self.n_symbols = n_symbols
        h = max(latent_dim*2, n_symbols)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, 2*n_symbols))
    def forward(self, z):
        X = self.net(z).view(-1, self.n_symbols, 2)
        pw = X.pow(2).sum(-1).mean(-1,keepdim=True).unsqueeze(-1)
        return X / (pw.sqrt() + 1e-8)


class JsccDecoder(nn.Module):
    """Received symbols Y -> received latent z'."""
    def __init__(self, latent_dim=768, n_symbols=2048):
        super().__init__()
        h = max(latent_dim*2, n_symbols)
        self.net = nn.Sequential(
            nn.Linear(2*n_symbols, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, latent_dim))
    def forward(self, Y):
        return self.net(Y.view(Y.size(0),-1))


class BottleneckDecoder(nn.Module):
    """Lightweight CNN reconstruction decoder: z' -> x_hat in [-1, 1].

    Works from the received latent only (no skip connections across the
    channel). The latent grid is resized to 7x7, projected to 768 channels and
    up-sampled 7 -> 14 -> 28 -> 56 -> 112 -> 224.
    """
    def __init__(self, patch_grid=8, patch_dim=16):
        super().__init__()
        self.patch_grid, self.patch_dim = patch_grid, patch_dim
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(patch_dim, 768, kernel_size=1),
            nn.GELU())
        def _up(i,o):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.GELU(),
                nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.GELU())
        self.up1=_up(768,384); self.up2=_up(384,192); self.up3=_up(192,96)
        self.up4=_up(96,48);   self.up5=_up(48,32)
        self.head = nn.Sequential(nn.Conv2d(32,3,1), nn.Tanh())

    def forward(self, z_prime_spatial):
        x = torch.nn.functional.interpolate(z_prime_spatial, size=(7, 7),
                                             mode="bilinear", align_corners=False)
        x = self.spatial_proj(x)                     # [B, 768, 7, 7]
        return self.head(self.up5(self.up4(self.up3(self.up2(self.up1(x))))))


class SwinJSCCSystem(nn.Module):
    """Swin-Tiny Deep JSCC system. forward(x) returns (x_hat, z, z_prime).

    n_symbols sets the compression ratio; set_channel() selects the channel
    model and SNR used by the next forward pass.
    """
    def __init__(self, n_symbols=2048, snr_db=10.0, channel_type="awgn",
                 pretrained=True, patch_grid=8, patch_dim=16, stage_idx=2):
        super().__init__()
        self.encoder  = SwinEncoder(pretrained=pretrained, patch_grid=patch_grid, patch_dim=patch_dim, stage_idx=stage_idx)
        ld = self.encoder.latent_dim
        self.mapper   = JsccMapper(ld, n_symbols)
        self.channel  = _make_channel(channel_type, snr_db)
        self.jscc_dec = JsccDecoder(ld, n_symbols)
        self.img_dec  = BottleneckDecoder(patch_grid=patch_grid, patch_dim=patch_dim)
        self.n_symbols = n_symbols; self.snr_db = snr_db
        self.patch_grid, self.patch_dim = patch_grid, patch_dim

    def set_channel(self, ctype, snr_db):
        dev = next(self.parameters()).device
        self.channel = _make_channel(ctype, snr_db).to(dev)
        self.snr_db  = snr_db

    def forward(self, x):
        z, _z_spatial_tx = self.encoder(x)
        X         = self.mapper(z)
        Y         = self.channel(X)
        z_prime   = self.jscc_dec(Y)
        z_prime_spatial = z_prime.view(-1, self.patch_dim, self.patch_grid, self.patch_grid)
        x_hat     = self.img_dec(z_prime_spatial)
        return x_hat, z, z_prime


class MaxLogitDetector(nn.Module):
    """Max Logit detector (Hendrycks et al., 2022): a linear classifier over the
    known classes applied to every latent cell; anomaly score = -max logit."""
    def __init__(self, patch_dim, patch_grid, num_classes):
        super().__init__()
        self.patch_dim, self.patch_grid, self.num_classes = patch_dim, patch_grid, num_classes
        self.classifier = nn.Linear(patch_dim, num_classes)

    def forward(self, z_prime):
        """Per-cell class logits [B, P, P, num_classes] from the flat received latent."""
        B = z_prime.size(0)
        p = z_prime.view(B, self.patch_dim, self.patch_grid, self.patch_grid)
        p = p.permute(0, 2, 3, 1).reshape(-1, self.patch_dim)
        logits = self.classifier(p)
        return logits.view(B, self.patch_grid, self.patch_grid, self.num_classes)

    @torch.no_grad()
    def anomaly_score(self, z_prime, img_size=224):
        """Pixel score map [B, 1, img_size, img_size] (each cell's score repeated over
        its pixels) and an image score (the highest cell score)."""
        logits = self.forward(z_prime)                     # [B, P, P, C]
        max_logit = logits.max(dim=-1).values               # [B, P, P]
        d = -max_logit                                       # higher = more anomalous
        img_score = d.reshape(d.size(0), -1).max(dim=1).values
        smap = torch.nn.functional.interpolate(
            d.unsqueeze(1), size=(img_size, img_size), mode="nearest")
        return smap, img_score


class PrototypeDetector(nn.Module):
    """Prototype detector (Fontanel et al., 2021): one prototype per known class,
    stored as the rows of a bias-free linear layer in the latent cell space.

    Training uses scaled cosine logits, scale * cos(cell, prototype); the anomaly
    score is the cosine distance to the nearest prototype, (1 - max cos) / 2.
    """

    def __init__(self, patch_dim, patch_grid, num_classes, scale):
        super().__init__()
        self.patch_dim, self.patch_grid, self.num_classes, self.scale = patch_dim, patch_grid, num_classes, scale
        self.classifier = nn.Linear(patch_dim, num_classes, bias=False)  # rows = prototypes

    def forward(self, z_prime):
        """Scaled cosine logits [B, P, P, num_classes], for the cross-entropy loss."""
        B = z_prime.size(0)
        p = z_prime.view(B, self.patch_dim, self.patch_grid, self.patch_grid)
        p = p.permute(0, 2, 3, 1).reshape(-1, self.patch_dim)
        p_norm = torch.nn.functional.normalize(p, dim=-1)
        w_norm = torch.nn.functional.normalize(self.classifier.weight, dim=-1)
        logits = self.scale * (p_norm @ w_norm.t())
        return logits.view(B, self.patch_grid, self.patch_grid, self.num_classes)

    @torch.no_grad()
    def anomaly_score(self, z_prime, img_size=224):
        """Pixel score map and image score, as for MaxLogitDetector."""
        B = z_prime.size(0)
        p = z_prime.view(B, self.patch_dim, self.patch_grid, self.patch_grid)
        p_flat = p.permute(0, 2, 3, 1).reshape(-1, self.patch_dim)
        p_norm = torch.nn.functional.normalize(p_flat, dim=-1)
        w_norm = torch.nn.functional.normalize(self.classifier.weight, dim=-1)  # [C, patch_dim]
        cos_sim = p_norm @ w_norm.t()                        # [B*P*P, C]
        max_cos = cos_sim.max(dim=-1).values
        d = (1.0 - max_cos) / 2.0                             # in [0, 1]
        d = d.view(B, self.patch_grid, self.patch_grid)
        img_score = d.reshape(B, -1).max(dim=1).values
        smap = torch.nn.functional.interpolate(
            d.unsqueeze(1), size=(img_size, img_size), mode="nearest")
        return smap, img_score


class PrototypeDetectorMagnitudeAware(PrototypeDetector):
    """Magnitude-aware scoring of a trained PrototypeDetector, used only for the
    scoring-rule check (dissertation Appendix F.5).

    Same weights as PrototypeDetector (load its state_dict; no retraining). The
    nearest-prototype cosine is multiplied by the cell's magnitude relative to the
    batch mean, so the score keeps the magnitude information that cosine scoring
    discards.
    """
    @torch.no_grad()
    def anomaly_score(self, z_prime, img_size=224):
        B = z_prime.size(0)
        p = z_prime.view(B, self.patch_dim, self.patch_grid, self.patch_grid)
        p_flat = p.permute(0, 2, 3, 1).reshape(-1, self.patch_dim)
        p_norm = torch.nn.functional.normalize(p_flat, dim=-1)
        w_norm = torch.nn.functional.normalize(self.classifier.weight, dim=-1)
        cos_sim = p_norm @ w_norm.t()
        max_cos = cos_sim.max(dim=-1).values
        magnitude = p_flat.norm(dim=-1)
        weighted_sim = max_cos * (magnitude / (magnitude.mean() + 1e-8))
        d = (1.0 - weighted_sim) / 2.0
        d = d.view(B, self.patch_grid, self.patch_grid)
        img_score = d.reshape(B, -1).max(dim=1).values
        smap = torch.nn.functional.interpolate(
            d.unsqueeze(1), size=(img_size, img_size), mode="nearest")
        return smap, img_score
