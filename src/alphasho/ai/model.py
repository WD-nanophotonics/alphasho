"""PyTorch policy-value network compatible with cshogi.dlshogi encoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from alphasho.ai.encoding import FEATURES1_NUM, FEATURES2_NUM
from alphasho.atomic_io import atomic_write
from alphasho.engine import POLICY_SIZE


@dataclass(frozen=True, slots=True)
class ModelConfig:
    channels: int = 128
    residual_blocks: int = 10
    value_hidden: int = 256

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> ModelConfig:
        values = values or {}
        return cls(
            channels=int(values.get("channels", 128)),
            residual_blocks=int(values.get("residual_blocks", 10)),
            value_hidden=int(values.get("value_hidden", 256)),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(inputs + hidden, inplace=True)


class ShogiPolicyValueNet(nn.Module):
    """Two-input dlshogi-style residual policy-value network."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        channels = config.channels
        self.board_conv = nn.Conv2d(
            FEATURES1_NUM, channels, 3, padding=1, bias=False
        )
        self.hand_conv = nn.Conv2d(FEATURES2_NUM, channels, 1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.tower = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(config.residual_blocks))
        )
        self.policy = nn.Conv2d(channels, POLICY_SIZE // 81, 1)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(81, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, features1: Tensor, features2: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.board_conv(features1) + self.hand_conv(features2)
        hidden = F.relu(self.input_bn(hidden), inplace=True)
        hidden = self.tower(hidden)
        logits = self.policy(hidden).flatten(1)
        value = F.relu(self.value_bn(self.value_conv(hidden)), inplace=True)
        value = F.relu(self.value_fc1(value.flatten(1)), inplace=True)
        value = torch.tanh(self.value_fc2(value)).flatten()
        return logits, value


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build has no CUDA support")
    return device


class TorchEvaluator:
    def __init__(self, model: ShogiPolicyValueNet, device: str = "auto") -> None:
        self.model = model
        self.device = resolve_device(device)
        self.model.to(self.device)

    def evaluate_batch(
        self,
        features1: NDArray[np.uint8],
        features2: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        inputs1 = torch.as_tensor(features1, dtype=torch.float32, device=self.device)
        inputs2 = torch.as_tensor(features2, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.inference_mode():
            logits, values = self.model(inputs1, inputs2)
        return (
            logits.detach().cpu().numpy().astype(np.float32, copy=False),
            values.detach().cpu().numpy().astype(np.float32, copy=False),
        )


def save_inference_checkpoint(
    path: str | Path,
    model: ShogiPolicyValueNet,
    *,
    search_config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    payload = {
        "schema_version": 1,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "search_config": dict(search_config or {}),
        "metadata": dict(metadata or {}),
    }
    atomic_write(
        destination,
        lambda temporary: torch.save(payload, temporary),
    )


def load_inference_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
) -> tuple[ShogiPolicyValueNet, TorchEvaluator, dict[str, Any]]:
    source = Path(path)
    checkpoint = torch.load(
        source,
        map_location=resolve_device(device),
        weights_only=True,
    )
    config = ModelConfig.from_dict(checkpoint.get("model_config"))
    model = ShogiPolicyValueNet(config)
    model.load_state_dict(checkpoint["model_state"])
    evaluator = TorchEvaluator(model, device=device)
    metadata = {
        "search_config": dict(checkpoint.get("search_config", {})),
        "metadata": dict(checkpoint.get("metadata", {})),
    }
    return model, evaluator, metadata
