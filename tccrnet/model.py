from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .resonant import ResonantEventBlock


class TemporalContext(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        heads = 4 if channels < 128 else 8
        if channels % heads:
            raise ValueError("context channels must be divisible by the attention heads")
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        normalized = self.norm1(sequence)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        sequence = sequence + attended
        sequence = sequence + self.feedforward(self.norm2(sequence))
        return sequence.transpose(1, 2)


class PyramidStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.AvgPool1d(2, stride=2),
        )
        self.context = nn.Sequential(
            TemporalContext(in_channels, dropout),
            nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat((self.local(x), self.context(x)), dim=1))


class GatedPyramidDecoderStage(nn.Module):
    """Restore one temporal scale through a globally conditioned analog skip."""

    def __init__(self, in_channels: int, skip_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, skip_channels, 4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(skip_channels),
            nn.SiLU(),
        )
        self.skip_filter = nn.Sequential(
            nn.Conv1d(
                skip_channels,
                skip_channels,
                5,
                padding=2,
                groups=skip_channels,
                bias=False,
            ),
            nn.BatchNorm1d(skip_channels),
            nn.SiLU(),
            nn.Conv1d(skip_channels, skip_channels, 1, bias=False),
        )
        self.route = nn.Conv1d(2 * skip_channels, skip_channels, 1)
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * skip_channels, skip_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(skip_channels),
            nn.SiLU(),
            nn.Conv1d(skip_channels, skip_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(skip_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.route.weight)
        nn.init.constant_(self.route.bias, -1.0)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        upsampled = self.up(x)
        if upsampled.shape[-1] != skip.shape[-1]:
            raise ValueError(
                "pyramid decoder and skip lengths differ: "
                f"{upsampled.shape[-1]} != {skip.shape[-1]}"
            )
        filtered_skip = self.skip_filter(skip)
        gate = torch.sigmoid(self.route(torch.cat((upsampled, skip), dim=1)))
        decoded = self.fusion(
            torch.cat((upsampled, gate * filtered_skip), dim=1)
        )
        return decoded, gate.mean()


class PyramidEventDenoiser(nn.Module):
    """Mid-capacity global synthesis network with a selective event bottleneck."""

    def __init__(
        self,
        base_channels: int = 16,
        context_channels: int = 64,
        signal_length: int = 512,
        sample_rate: float = 256.0,
        spiking: bool = False,
        dropout: float = 0.1,
        recurrent_rank: int | None = None,
        decoder_mode: str = "global",
        stage_count: int = 5,
    ) -> None:
        super().__init__()
        if base_channels < 8 or base_channels % 4:
            raise ValueError("base_channels must be at least eight and divisible by four")
        if context_channels < 8:
            raise ValueError("context_channels must be at least eight")
        if stage_count not in (4, 5):
            raise ValueError("stage_count must be four or five")
        compression_factor = 2 ** (stage_count + 1)
        if signal_length < compression_factor or signal_length % compression_factor:
            raise ValueError(
                f"signal_length must be divisible by {compression_factor}"
            )
        if decoder_mode not in ("global", "gated-fusion", "gated-dilated"):
            raise ValueError(
                "decoder_mode must be 'global', 'gated-fusion', or 'gated-dilated'"
            )
        self.base_channels = int(base_channels)
        self.context_channels = int(context_channels)
        self.signal_length = int(signal_length)
        self.sample_rate = float(sample_rate)
        self.spiking = bool(spiking)
        self.dropout = float(dropout)
        self.recurrent_rank = recurrent_rank
        self.decoder_mode = decoder_mode
        self.stage_count = int(stage_count)
        self.compression_factor = compression_factor
        widths = tuple(base_channels * (2**index) for index in range(stage_count + 1))
        self.widths = widths

        self.stem = nn.Sequential(
            nn.Conv1d(1, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.SiLU(),
            nn.Conv1d(widths[0], widths[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.SiLU(),
            nn.AvgPool1d(2, stride=2),
        )
        self.stages = nn.Sequential(
            *[
                PyramidStage(widths[index], widths[index + 1], dropout)
                for index in range(len(widths) - 1)
            ]
        )
        latent_channels = widths[-1]
        self.final_local = nn.Sequential(
            nn.Conv1d(latent_channels, latent_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(latent_channels),
            nn.SiLU(),
            nn.Conv1d(latent_channels, latent_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(latent_channels),
            nn.SiLU(),
        )
        self.event_in = nn.Conv1d(latent_channels, context_channels, 1)
        self.event = ResonantEventBlock(
            context_channels,
            latent_sample_rate=sample_rate / compression_factor,
            dilation=1,
            recurrent_rank=recurrent_rank,
            spiking=spiking,
            residual_gain_init=-0.5,
            dropout=dropout,
        )
        self.event_out = nn.Conv1d(context_channels, latent_channels, 1)
        self.raw_event_gain = nn.Parameter(torch.tensor(-1.5))
        latent_length = signal_length // compression_factor
        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(latent_channels * latent_length, signal_length),
        )
        if decoder_mode in ("gated-fusion", "gated-dilated"):
            self.decoder = nn.ModuleList(
                [
                    GatedPyramidDecoderStage(in_channels, skip_channels, dropout)
                    for in_channels, skip_channels in zip(
                        reversed(widths[1:]), reversed(widths[:-1])
                    )
                ]
            )
            self.output_up = nn.Sequential(
                nn.ConvTranspose1d(
                    widths[0], widths[0], 4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm1d(widths[0]),
                nn.SiLU(),
                nn.Conv1d(widths[0], widths[0], 7, padding=3, bias=False),
                nn.BatchNorm1d(widths[0]),
                nn.SiLU(),
            )
            self.artifact_head = nn.Conv1d(widths[0], 1, 9, padding=4)
            self.local_router = nn.Sequential(
                nn.Conv1d(3, widths[0], 9, padding=4),
                nn.SiLU(),
                nn.Conv1d(widths[0], 1, 9, padding=4),
            )
            self.raw_artifact_scale = nn.Parameter(torch.tensor(-1.0))
            self.raw_local_mix = nn.Parameter(torch.tensor(-1.0))
            nn.init.zeros_(self.artifact_head.weight)
            nn.init.zeros_(self.artifact_head.bias)
            nn.init.zeros_(self.local_router[-1].weight)
            nn.init.zeros_(self.local_router[-1].bias)
            self.waveform_refiner = (
                DilatedWaveformRefiner(widths[0], dropout)
                if decoder_mode == "gated-dilated"
                else None
            )
        else:
            self.decoder = None
            self.output_up = None
            self.artifact_head = None
            self.local_router = None
            self.register_parameter("raw_artifact_scale", None)
            self.register_parameter("raw_local_mix", None)
            self.waveform_refiner = None

    def forward(self, x: torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"expected [batch, 1, time], got {tuple(x.shape)}")
        if x.shape[-1] != self.signal_length:
            raise ValueError(
                f"expected signal length {self.signal_length}, got {x.shape[-1]}"
            )
        features = [self.stem(x)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        latent = self.final_local(features[-1])
        events = self.event_in(latent)
        events, event_stats = self.event(events)
        event_gain = torch.sigmoid(self.raw_event_gain)
        latent = latent + event_gain * self.event_out(events)
        global_clean = self.readout(latent)
        stats = {
            "block_0_rate": event_stats["rate"],
            "block_0_spikes": event_stats["spikes"],
            "block_0_radius": event_stats["mean_radius"],
            "block_0_frequency_hz": event_stats["mean_frequency_hz"],
            "mean_rate": event_stats["rate"],
            "event_gain": event_gain,
        }
        if self.decoder_mode in ("gated-fusion", "gated-dilated"):
            assert self.decoder is not None
            assert self.output_up is not None
            assert self.artifact_head is not None
            assert self.local_router is not None
            decoded = latent
            for index, (decoder, skip) in enumerate(
                zip(self.decoder, reversed(features[:-1]))
            ):
                decoded, gate_mean = decoder(decoded, skip)
                stats[f"skip_{index}_gate"] = gate_mean
            decoded = self.output_up(decoded)
            artifact = self.artifact_head(decoded)[:, 0]
            artifact_scale = torch.nn.functional.softplus(self.raw_artifact_scale)
            local_clean = x[:, 0] - artifact_scale * artifact
            local_route = torch.sigmoid(
                self.local_router(
                    torch.stack((x[:, 0], global_clean, local_clean), dim=1)
                )
            )[:, 0]
            local_mix = torch.sigmoid(self.raw_local_mix) * local_route
            denoised = global_clean + local_mix * (local_clean - global_clean)
            stats["artifact_scale"] = artifact_scale
            stats["local_mix"] = local_mix.mean()
            if self.waveform_refiner is not None:
                denoised, correction = self.waveform_refiner(x[:, 0], denoised)
                stats["refiner_correction_rms"] = correction.square().mean().sqrt()
        else:
            denoised = global_clean
        return denoised, stats

    def config(self) -> Dict[str, object]:
        return {
            "base_channels": self.base_channels,
            "context_channels": self.context_channels,
            "signal_length": self.signal_length,
            "sample_rate": self.sample_rate,
            "spiking": self.spiking,
            "dropout": self.dropout,
            "recurrent_rank": self.recurrent_rank,
            "decoder_mode": self.decoder_mode,
            "stage_count": self.stage_count,
        }


class TCCRNet(PyramidEventDenoiser):
    """Five-stage gated TCCR-Net with the formal ECG model defaults.

    Returns a pair: clean waveform [batch, time] and diagnostic tensors.
    No preprocessing, pretrained weights, or persistent streaming state.
    """

    def __init__(
        self,
        base_channels: int = 8,
        context_channels: int = 32,
        signal_length: int = 512,
        sample_rate: float = 256.0,
        spiking: bool = False,
        dropout: float = 0.1,
        recurrent_rank: int | None = None,
        decoder_mode: str = "gated-fusion",
        stage_count: int = 5,
    ) -> None:
        if decoder_mode != "gated-fusion" or stage_count != 5:
            raise ValueError("TCCR-Net requires five stages and gated-fusion decoding")
        super().__init__(
            base_channels=base_channels,
            context_channels=context_channels,
            signal_length=signal_length,
            sample_rate=sample_rate,
            spiking=spiking,
            dropout=dropout,
            recurrent_rank=recurrent_rank,
            decoder_mode=decoder_mode,
            stage_count=stage_count,
        )
