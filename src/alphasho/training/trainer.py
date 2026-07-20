"""Policy-value optimization and resumable training checkpoints."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from alphasho.ai import ModelConfig, SearchCancelled, ShogiPolicyValueNet
from alphasho.ai.model import resolve_device
from alphasho.atomic_io import atomic_write
from alphasho.engine import POLICY_SIZE
from alphasho.training.data import TrainingSample


@dataclass(frozen=True, slots=True)
class TrainMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    steps: int
    target_entropy: float = 0.0
    kl: float = 0.0
    gradient_norm: float = 0.0
    samples_per_second: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainStepMetrics:
    step: int
    total_steps: int
    loss: float
    policy_loss: float
    value_loss: float
    predicted_entropy: float
    target_entropy: float
    kl: float
    gradient_norm: float
    learning_rate: float
    samples_per_second: float
    elapsed_seconds: float


class PolicyValueTrainer:
    def __init__(
        self,
        model: ShogiPolicyValueNet,
        *,
        device: str,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self.model = model
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train(
        self,
        samples: list[TrainingSample],
        *,
        steps: int,
        batch_size: int,
        seed: int,
        start_step: int = 0,
        log_every_steps: int = 10,
        progress_callback: Callable[[TrainStepMetrics], None] | None = None,
        step_callback: Callable[[int], None] | None = None,
        checkpoint_callback: Callable[[int], None] | None = None,
        checkpoint_every_steps: int = 25,
        stop_event: threading.Event | None = None,
    ) -> TrainMetrics:
        if not samples:
            raise ValueError("Training requires at least one replay sample")
        if start_step >= steps:
            return TrainMetrics(0.0, 0.0, 0.0, 0.0, 0)
        totals = np.zeros(6, dtype=np.float64)
        last_kl = last_gradient = 0.0
        started = time.monotonic()
        completed = 0
        self.model.train()
        for absolute_step in range(start_step, steps):
            if stop_event is not None and stop_event.is_set():
                raise SearchCancelled("Network training was stopped")
            step_started = time.monotonic()
            step_rng = np.random.default_rng(seed + absolute_step)
            indices = step_rng.integers(0, len(samples), size=batch_size)
            batch = [samples[int(index)] for index in indices]
            features1 = torch.as_tensor(
                np.stack([item.features1 for item in batch]),
                dtype=torch.float32,
                device=self.device,
            )
            features2 = torch.as_tensor(
                np.stack([item.features2 for item in batch]),
                dtype=torch.float32,
                device=self.device,
            )
            values = torch.as_tensor(
                [item.value for item in batch],
                dtype=torch.float32,
                device=self.device,
            )
            targets = torch.zeros(
                (len(batch), POLICY_SIZE),
                dtype=torch.float32,
                device=self.device,
            )
            for row, sample in enumerate(batch):
                labels = torch.as_tensor(
                    sample.policy_labels.astype(np.int64),
                    device=self.device,
                )
                probabilities = torch.as_tensor(
                    sample.policy_probabilities.astype(np.float32),
                    device=self.device,
                )
                targets[row, labels] = probabilities
            self.optimizer.zero_grad(set_to_none=True)
            logits, predicted_values = self.model(features1, features2)
            log_probabilities = F.log_softmax(logits, dim=1)
            policy_loss = -(targets * log_probabilities).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted_values, values)
            loss = policy_loss + value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at step {absolute_step + 1}: {loss}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 5.0
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    "Non-finite gradient norm at step "
                    f"{absolute_step + 1}: {gradient_norm}"
                )
            self.optimizer.step()
            completed_step = absolute_step + 1
            if step_callback is not None:
                step_callback(completed_step)
            with torch.no_grad():
                probabilities = log_probabilities.exp()
                entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
                target_log = torch.log(targets.clamp_min(1e-12))
                target_entropy = -(targets * target_log).sum(dim=1).mean()
            completed += 1
            step_seconds = time.monotonic() - step_started
            samples_per_second = batch_size / max(step_seconds, 1e-9)
            should_report = (
                completed_step % max(1, log_every_steps) == 0
                or completed_step == steps
            )
            kl = 0.0
            if should_report:
                self.model.eval()
                with torch.no_grad():
                    updated_logits, _updated_values = self.model(features1, features2)
                    updated_log = F.log_softmax(updated_logits, dim=1)
                    kl = float(
                        (
                            probabilities
                            * (log_probabilities.detach() - updated_log)
                        )
                        .sum(dim=1)
                        .mean()
                    )
                self.model.train()
            last_kl = kl if should_report else last_kl
            last_gradient = float(gradient_norm)
            totals += (
                float(loss.detach()),
                float(policy_loss.detach()),
                float(value_loss.detach()),
                float(entropy.detach()),
                float(target_entropy.detach()),
                samples_per_second,
            )
            if should_report and progress_callback is not None:
                progress_callback(
                    TrainStepMetrics(
                        completed_step,
                        steps,
                        float(loss.detach()),
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float(entropy.detach()),
                        float(target_entropy.detach()),
                        kl,
                        float(gradient_norm),
                        float(self.optimizer.param_groups[0]["lr"]),
                        samples_per_second,
                        time.monotonic() - started,
                    )
                )
            if (
                checkpoint_callback is not None
                and completed_step % max(1, checkpoint_every_steps) == 0
            ):
                checkpoint_callback(completed_step)
        totals /= completed
        return TrainMetrics(
            loss=float(totals[0]),
            policy_loss=float(totals[1]),
            value_loss=float(totals[2]),
            entropy=float(totals[3]),
            steps=completed,
            target_entropy=float(totals[4]),
            kl=last_kl,
            gradient_norm=last_gradient,
            samples_per_second=float(totals[5]),
        )

    def save_state(
        self,
        path: str | Path,
        *,
        iteration: int,
        metadata: dict[str, Any],
    ) -> None:
        destination = Path(path)
        payload = {
            "schema_version": 1,
            "iteration": iteration,
            "model_config": self.model.config.to_dict(),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "metadata": metadata,
        }
        atomic_write(
            destination,
            lambda temporary: torch.save(payload, temporary),
        )

    def load_state(
        self,
        path: str | Path,
        *,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
    ) -> tuple[int, dict[str, Any]]:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        checkpoint_config = ModelConfig.from_dict(checkpoint.get("model_config"))
        if checkpoint_config != self.model.config:
            raise ValueError("Training checkpoint model configuration does not match")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        for group in self.optimizer.param_groups:
            if learning_rate is not None:
                group["lr"] = learning_rate
            if weight_decay is not None:
                group["weight_decay"] = weight_decay
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.random.set_rng_state(checkpoint["torch_random_state"])
        return int(checkpoint.get("iteration", 0)), dict(checkpoint.get("metadata", {}))
