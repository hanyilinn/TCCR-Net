from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spike import _inverse_softplus, surrogate_spike


def _logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(value) - torch.log1p(-value)


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class BandResonantSpikingUnit(nn.Module):
    """Stable complex-pole resonator with event output.

    One downsampled latent frame is one state update.  Channels start at EEG
    band-centre frequencies and learn both pole angle and input-dependent pole
    radius.  This supplies oscillatory memory without Poisson/rate replication.
    """

    def __init__(
        self,
        channels: int,
        latent_sample_rate: float = 32.0,
        recurrent_rank: int | None = None,
        threshold: float = 0.6,
        spiking: bool = True,
        spike_amplitude_init: float = 1.0,
        spike_mode: str = "positive",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.latent_sample_rate = float(latent_sample_rate)
        self.rank = recurrent_rank or max(8, channels // 8)
        self.spiking = spiking
        if spike_mode not in ("positive", "signed"):
            raise ValueError("spike_mode must be 'positive' or 'signed'")
        self.spike_mode = spike_mode
        if spike_amplitude_init <= 0:
            raise ValueError("spike_amplitude_init must be positive")
        self.input_projection = nn.Conv1d(channels, 2 * channels, 1)
        self.recurrent_down = nn.Linear(channels, self.rank, bias=False)
        self.recurrent_up = nn.Linear(self.rank, channels, bias=False)

        nyquist = self.latent_sample_rate / 2.0
        canonical = torch.tensor((0.75, 2.0, 5.0, 9.0, 13.0, 0.82 * nyquist))
        canonical = canonical.clamp(max=0.92 * nyquist)
        frequencies = canonical.repeat(math.ceil(channels / len(canonical)))[:channels]
        angles = (2.0 * math.pi * frequencies / self.latent_sample_rate).clamp(
            1e-3, math.pi - 1e-3
        )
        self.raw_angle = nn.Parameter(_logit(angles / math.pi))
        base_radius = torch.full((channels,), 0.94)
        self.raw_radius = nn.Parameter(_logit((base_radius - 0.5) / 0.49))
        self.raw_threshold = nn.Parameter(
            torch.full((channels,), _inverse_softplus(threshold))
        )
        if spiking:
            self.raw_spike_amplitude = nn.Parameter(
                torch.full((channels,), _inverse_softplus(spike_amplitude_init))
            )
        else:
            self.register_parameter("raw_spike_amplitude", None)
        self.drive_gain = nn.Parameter(torch.full((channels,), 0.35))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.35)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.orthogonal_(self.recurrent_down.weight, gain=0.25)
        nn.init.xavier_uniform_(self.recurrent_up.weight, gain=0.25)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected [batch, {self.channels}, time], got {tuple(x.shape)}"
            )
        batch, channels, steps = x.shape
        drive_sequence, radius_modulation = self.input_projection(x).chunk(2, dim=1)
        real = x.new_zeros(batch, channels)
        imaginary = x.new_zeros(batch, channels)
        events = x.new_zeros(batch, channels)
        angle = math.pi * torch.sigmoid(self.raw_angle).unsqueeze(0)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        base_radius = self.raw_radius.unsqueeze(0)
        threshold = F.softplus(self.raw_threshold).unsqueeze(0) + 1e-4
        spike_amplitude = (
            F.softplus(self.raw_spike_amplitude).unsqueeze(0)
            if self.raw_spike_amplitude is not None
            else None
        )
        gain = F.softplus(self.drive_gain).unsqueeze(0)
        event_steps = []
        binary_steps = []
        radius_steps = []

        for index in range(steps):
            recurrent = self.recurrent_up(self.recurrent_down(events))
            radius = 0.5 + 0.49 * torch.sigmoid(
                base_radius + 0.25 * torch.tanh(radius_modulation[:, :, index])
            )
            previous_real = real
            real = radius * (cosine * real - sine * imaginary)
            imaginary = radius * (sine * previous_real + cosine * imaginary)
            real = real + gain * (drive_sequence[:, :, index] + recurrent)
            if self.spiking:
                positive_events = surrogate_spike(real - threshold, width=1.0)
                if self.spike_mode == "signed":
                    negative_events = surrogate_spike(
                        -real - threshold, width=1.0
                    )
                    binary_events = positive_events - negative_events
                else:
                    binary_events = positive_events
                real = real - binary_events.detach() * threshold
                events = binary_events * spike_amplitude
                binary_steps.append(binary_events)
            else:
                events = F.silu(real)
            event_steps.append(events)
            radius_steps.append(radius)

        sequence = torch.stack(event_steps, dim=-1)
        radii = torch.stack(radius_steps, dim=-1)
        if binary_steps:
            binary_sequence = torch.stack(binary_steps, dim=-1)
            rate = binary_sequence.abs().mean()
            spikes = binary_sequence.abs().sum()
        else:
            rate = sequence.abs().mean()
            spikes = sequence.abs().sum()
        return sequence, {
            "rate": rate,
            "spikes": spikes,
            "mean_event_scale": (
                spike_amplitude.mean()
                if spike_amplitude is not None
                else sequence.new_ones(())
            ),
            "mean_radius": radii.mean(),
            "mean_frequency_hz": (
                self.latent_sample_rate * angle.mean() / (2.0 * math.pi)
            ),
        }


class ResonantEventBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        latent_sample_rate: float,
        dilation: int,
        recurrent_rank: int | None = None,
        spiking: bool = True,
        residual_gain_init: float = -1.5,
        dropout: float = 0.0,
        normalization: str = "group",
        spike_amplitude_init: float = 1.0,
        spike_mode: str = "positive",
    ) -> None:
        super().__init__()
        if normalization == "group":
            self.norm = nn.GroupNorm(_groups(channels), channels)
            self.output_norm = nn.GroupNorm(_groups(channels), channels)
        elif normalization == "batch":
            self.norm = nn.BatchNorm1d(channels)
            self.output_norm = nn.BatchNorm1d(channels)
        else:
            raise ValueError("normalization must be 'group' or 'batch'")
        self.neuron = BandResonantSpikingUnit(
            channels,
            latent_sample_rate=latent_sample_rate,
            recurrent_rank=recurrent_rank,
            spiking=spiking,
            spike_amplitude_init=spike_amplitude_init,
            spike_mode=spike_mode,
        )
        self.event_depthwise = nn.Conv1d(
            channels,
            channels,
            5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.event_project = nn.Conv1d(channels, channels, 1, bias=False)
        self.route = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.residual_gain = nn.Parameter(torch.tensor(residual_gain_init))

    def forward(self, x: torch.Tensor):
        events, neuron_stats = self.neuron(self.norm(x))
        update = self.dropout(
            self.output_norm(self.event_project(self.event_depthwise(events)))
        )
        route = torch.sigmoid(self.route(x))
        output = x + torch.sigmoid(self.residual_gain) * route * update
        return output, neuron_stats
