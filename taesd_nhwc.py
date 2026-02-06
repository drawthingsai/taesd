#!/usr/bin/env python3
"""
NHWC-wrapped TAESD modules.
All activations are NHWC; each op that requires NCHW internally permutes in/out.
Weights remain in standard NCHW (OIHW) layout.
"""
import torch
import torch.nn as nn


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def _to_nhwc(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 1)


class NHWCConv2d(nn.Conv2d):
    def forward(self, x):
        x = _to_nchw(x)
        y = super().forward(x)
        return _to_nhwc(y)


class NHWCGroupNorm(nn.GroupNorm):
    def forward(self, x):
        x = _to_nchw(x)
        y = super().forward(x)
        return _to_nhwc(y)


class NHWCUpsample(nn.Upsample):
    def forward(self, x):
        x = _to_nchw(x)
        y = super().forward(x)
        return _to_nhwc(y)


def conv(n_in, n_out, **kwargs):
    return NHWCConv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out), nn.ReLU(),
            conv(n_out, n_out), nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = NHWCConv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1 = lambda n_in, n_out: NHWCConv2d(n_in, n_out, 1, bias=False)
            n_gn = n_in * 4
            self.pool = nn.Sequential(
                conv1x1(n_in, n_gn),
                NHWCGroupNorm(4, n_gn),
                nn.ReLU(inplace=True),
                conv1x1(n_gn, n_in),
            )

    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


def Encoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw),
        conv(64, latent_channels),
    )


def Decoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw),
        NHWCUpsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64),
        NHWCUpsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64),
        NHWCUpsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


def F32Encoder(latent_channels=32):
    return nn.Sequential(
        conv(3, 32, stride=2), nn.ReLU(inplace=True), conv(32, 64, stride=2), nn.ReLU(inplace=True), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )


def F32Decoder(latent_channels=32):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), NHWCUpsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), NHWCUpsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), NHWCUpsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), NHWCUpsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), NHWCUpsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", decoder_path="taesd_decoder.pth", latent_channels=None, arch_variant=None):
        super().__init__()
        if latent_channels is None:
            latent_channels, arch_variant = self.guess_latent_channels_and_arch(str(encoder_path))
        self.encoder = Encoder(latent_channels, use_midblock_gn=(arch_variant in ["flux_2"]))
        self.decoder = Decoder(latent_channels, use_midblock_gn=(arch_variant in ["flux_2"]))
        if arch_variant == "f32":
            self.encoder, self.decoder = F32Encoder(latent_channels), F32Decoder(latent_channels)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))

    def guess_latent_channels(self, encoder_path):
        return self.guess_latent_channels_and_arch(encoder_path)[0]

    def guess_latent_channels_and_arch(self, encoder_path):
        if "taef1" in encoder_path:
            return 16, None
        if "taef2" in encoder_path:
            return 32, "flux_2"
        if "taesd3" in encoder_path:
            return 16, None
        if "taesana" in encoder_path:
            return 32, "f32"
        return 4, None

    @staticmethod
    def scale_latents(x):
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)


@torch.no_grad()
def main():
    from PIL import Image
    import sys
    import torchvision.transforms.functional as TF
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device", dev)
    taesd = TAESD().to(dev)
    for im_path in sys.argv[1:]:
        im = TF.to_tensor(Image.open(im_path).convert("RGB")).unsqueeze(0).to(dev)
        # convert to NHWC for encoder
        im = im.permute(0, 2, 3, 1)

        im_enc = taesd.scale_latents(taesd.encoder(im)).mul_(255).round_().byte()
        enc_path = im_path + ".encoded.png"
        TF.to_pil_image(im_enc[0].permute(2, 0, 1)).save(enc_path)
        print(f"Encoded {im_path} to {enc_path}")

        im_enc = taesd.unscale_latents(TF.to_tensor(Image.open(enc_path)).unsqueeze(0).to(dev))
        im_enc = im_enc.permute(0, 2, 3, 1)
        im_dec = taesd.decoder(im_enc).clamp(0, 1)
        dec_path = im_path + ".decoded.png"
        print(f"Decoded {enc_path} to {dec_path}")
        TF.to_pil_image(im_dec[0].permute(2, 0, 1)).save(dec_path)


if __name__ == "__main__":
    main()
