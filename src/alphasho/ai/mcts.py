"""PUCT Monte Carlo tree search for shogi."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from alphasho.ai.encoding import encode_position, normalize_legal_logits
from alphasho.engine import SearchablePosition, SearchAction


class SearchCancelled(RuntimeError):
    pass


class BatchEvaluator(Protocol):
    def evaluate_batch(
        self,
        features1: NDArray[np.uint8],
        features2: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        ...


@dataclass(frozen=True, slots=True)
class SearchConfig:
    n_playouts: int = 400
    c_puct: float = 1.5
    batch_size: int = 16
    virtual_loss: float = 1.0
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> SearchConfig:
        values = values or {}
        return cls(
            n_playouts=int(values.get("n_playouts", 400)),
            c_puct=float(values.get("c_puct", 1.5)),
            batch_size=int(values.get("batch_size", 16)),
            virtual_loss=float(values.get("virtual_loss", 1.0)),
            dirichlet_alpha=float(
                values.get("dirichlet_alpha", 0.15)
            ),
            dirichlet_epsilon=float(
                values.get("dirichlet_epsilon", 0.25)
            ),
        )

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TreeNode:
    __slots__ = (
        "action",
        "prior",
        "children",
        "visits",
        "value_sum",
        "virtual_visits",
    )

    def __init__(self, action: SearchAction | None, prior: float) -> None:
        self.action = action
        self.prior = float(prior)
        self.children: dict[int, TreeNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.virtual_visits = 0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def expand(
        self,
        actions: tuple[SearchAction, ...],
        probabilities: NDArray[np.float32],
    ) -> int:
        created = 0
        for action, probability in zip(actions, probabilities, strict=True):
            if action.label not in self.children:
                self.children[action.label] = TreeNode(action, float(probability))
                created += 1
        return created

    def select(self, c_puct: float, virtual_loss: float) -> TreeNode:
        parent_scale = math.sqrt(self.visits + self.virtual_visits + 1.0)
        return max(
            self.children.values(),
            key=lambda child: (
                -child.q_value
                + c_puct
                * child.prior
                * parent_scale
                / (1 + child.visits + child.virtual_visits)
                - virtual_loss * child.virtual_visits
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchProgress:
    completed_playouts: int
    total_playouts: int
    elapsed_seconds: float
    new_nodes: int
    evaluation_batches: int
    evaluated_leaves: int
    max_depth: int = 0
    root_value: float = 0.0
    principal_variation: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchStats:
    playouts: int
    elapsed_seconds: float
    new_nodes: int
    evaluation_batches: int
    evaluated_leaves: int
    max_depth: int = 0

    @property
    def playouts_per_second(self) -> float:
        return self.playouts / max(self.elapsed_seconds, 1e-9)

    @property
    def average_batch_size(self) -> float:
        return self.evaluated_leaves / max(self.evaluation_batches, 1)


@dataclass(frozen=True, slots=True)
class SearchResult:
    action_usis: tuple[str, ...]
    action_labels: NDArray[np.int32]
    probabilities: NDArray[np.float32]
    visits: NDArray[np.int32]
    root_value: float
    principal_variation: tuple[str, ...]
    stats: SearchStats

    def choose(self, rng: np.random.Generator) -> str:
        if not self.action_usis:
            raise ValueError("Search produced no legal action")
        index = int(rng.choice(len(self.action_usis), p=self.probabilities))
        return self.action_usis[index]


@dataclass(slots=True)
class _PendingLeaf:
    node: TreeNode
    path: list[TreeNode]
    position: SearchablePosition


class MCTS:
    def __init__(
        self,
        evaluator: BatchEvaluator,
        config: SearchConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(seed)
        self.root = TreeNode(None, 1.0)
        self._root_key: tuple[object, ...] | None = None

    def reset(self) -> None:
        self.root = TreeNode(None, 1.0)
        self._root_key = None

    def advance(
        self,
        usi: str,
        position: SearchablePosition | None = None,
    ) -> None:
        matching = next(
            (
                child
                for child in self.root.children.values()
                if child.action is not None and child.action.usi == usi
            ),
            None,
        )
        self.root = matching if matching is not None else TreeNode(None, 1.0)
        self.root.action = None
        self._root_key = self._key(position) if position is not None else None

    def search(
        self,
        position: SearchablePosition,
        *,
        temperature: float = 1e-3,
        add_root_noise: bool = False,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[SearchProgress], None] | None = None,
        heartbeat_seconds: float = 10.0,
        min_playouts: int = 0,
        max_playouts: int | None = None,
        deadline: float | None = None,
        max_depth: int | None = None,
    ) -> SearchResult:
        started = time.monotonic()
        last_progress = started
        new_nodes = evaluation_batches = evaluated_leaves = 0
        deepest = 0
        playout_limit = max(
            1,
            self.config.n_playouts
            if max_playouts is None
            else int(max_playouts),
        )
        minimum = min(max(0, int(min_playouts)), playout_limit)
        terminal = position.terminal_value()
        if terminal is not None:
            raise ValueError("Cannot search a terminal position")
        self._synchronize(position)
        if not self.root.children:
            initial_leaf = _PendingLeaf(self.root, [self.root], position.clone())
            self._reserve(initial_leaf.path)
            new_nodes += self._evaluate_and_expand([initial_leaf])
            evaluation_batches += 1
            evaluated_leaves += 1
        if add_root_noise:
            self._add_root_noise()

        completed = 0
        while completed < playout_limit:
            self._check_cancelled(cancel_event)
            if self._deadline_reached(deadline):
                break
            pending: list[_PendingLeaf] = []
            target = min(
                self.config.batch_size,
                playout_limit - completed,
            )
            while len(pending) < target and completed < playout_limit:
                if self._deadline_reached(deadline):
                    break
                leaf = self._select_leaf(position, max_depth=max_depth)
                deepest = max(deepest, len(leaf.path) - 1)
                value = leaf.position.terminal_value()
                if value is None:
                    self._reserve(leaf.path)
                    pending.append(leaf)
                else:
                    self._backup(leaf.path, value)
                    completed += 1
            if pending:
                self._check_cancelled(cancel_event)
                new_nodes += self._evaluate_and_expand(pending)
                evaluation_batches += 1
                evaluated_leaves += len(pending)
                completed += len(pending)
            elif self._deadline_reached(deadline):
                break
            now = time.monotonic()
            if (
                progress_callback is not None
                and now - last_progress >= heartbeat_seconds
            ):
                progress_callback(
                    SearchProgress(
                        completed,
                        playout_limit,
                        now - started,
                        new_nodes,
                        evaluation_batches,
                        evaluated_leaves,
                        deepest,
                        float(self.root.q_value),
                        self._principal_variation(),
                    )
                )
                last_progress = now

            # ``minimum`` is a desired floor only. A hard time deadline always
            # wins, which prevents a slow device from losing on time.
            if completed < minimum and not self._deadline_reached(deadline):
                continue

        children = tuple(self.root.children.values())
        if not children:
            raise ValueError("Search position has no legal moves")
        counts = np.asarray([child.visits for child in children], dtype=np.float64)
        if not np.any(counts):
            counts = np.asarray([child.prior for child in children], dtype=np.float64)
        probabilities = self._temperature_probabilities(counts, temperature)
        visits = np.asarray([child.visits for child in children], dtype=np.int32)
        root_value = self.root.q_value
        elapsed = time.monotonic() - started
        return SearchResult(
            action_usis=tuple(child.action.usi for child in children if child.action),
            action_labels=np.asarray(
                [child.action.label for child in children if child.action],
                dtype=np.int32,
            ),
            probabilities=probabilities,
            visits=visits,
            root_value=float(root_value),
            principal_variation=self._principal_variation(),
            stats=SearchStats(
                completed,
                elapsed,
                new_nodes,
                evaluation_batches,
                evaluated_leaves,
                deepest,
            ),
        )

    def _synchronize(self, position: SearchablePosition) -> None:
        key = self._key(position)
        if self._root_key != key:
            self.root = TreeNode(None, 1.0)
            self._root_key = key

    @staticmethod
    def _key(position: SearchablePosition | None) -> tuple[object, ...] | None:
        return None if position is None else position.state_key

    def _select_leaf(
        self,
        root_position: SearchablePosition,
        *,
        max_depth: int | None = None,
    ) -> _PendingLeaf:
        position = root_position.clone()
        node = self.root
        path = [node]
        while node.children and (
            max_depth is None or len(path) - 1 < max_depth
        ):
            node = node.select(self.config.c_puct, self.config.virtual_loss)
            if node.action is None:
                raise RuntimeError("Non-root tree node is missing its action")
            position.push(node.action)
            path.append(node)
            if position.terminal_value() is not None:
                break
        return _PendingLeaf(node, path, position)

    def _evaluate_and_expand(self, leaves: list[_PendingLeaf]) -> int:
        encoded = [encode_position(leaf.position) for leaf in leaves]
        features1 = np.stack([item.features1 for item in encoded])
        features2 = np.stack([item.features2 for item in encoded])
        try:
            logits, values = self.evaluator.evaluate_batch(features1, features2)
        except BaseException:
            for leaf in leaves:
                self._release(leaf.path)
            raise
        if logits.shape != (len(leaves), 2187) or values.shape != (len(leaves),):
            for leaf in leaves:
                self._release(leaf.path)
            raise ValueError(
                "Evaluator returned invalid shapes "
                f"{logits.shape} and {values.shape}"
            )
        created = 0
        for index, leaf in enumerate(leaves):
            self._release(leaf.path)
            actions = leaf.position.legal_actions()
            labels = np.asarray([action.label for action in actions], dtype=np.int32)
            probabilities = normalize_legal_logits(logits[index], labels)
            created += leaf.node.expand(actions, probabilities)
            self._backup(leaf.path, float(values[index]))
        return created

    @staticmethod
    def _reserve(path: list[TreeNode]) -> None:
        for node in path:
            node.virtual_visits += 1

    @staticmethod
    def _release(path: list[TreeNode]) -> None:
        for node in path:
            node.virtual_visits -= 1

    @staticmethod
    def _backup(path: list[TreeNode], leaf_value: float) -> None:
        value = float(leaf_value)
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _add_root_noise(self) -> None:
        children = tuple(self.root.children.values())
        if not children:
            return
        alpha = self.config.dirichlet_alpha
        epsilon = self.config.dirichlet_epsilon
        if alpha <= 0 or epsilon <= 0:
            return
        noise = self.rng.dirichlet(np.full(len(children), alpha))
        for child, sample in zip(children, noise, strict=True):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * float(sample)

    @staticmethod
    def _temperature_probabilities(
        counts: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float32]:
        if temperature <= 1e-6:
            result = np.zeros(len(counts), dtype=np.float32)
            result[int(np.argmax(counts))] = 1.0
            return result
        scaled = np.log(counts + 1e-12) / temperature
        scaled -= np.max(scaled)
        result = np.exp(scaled)
        result /= result.sum()
        return result.astype(np.float32)

    def _principal_variation(self) -> tuple[str, ...]:
        moves: list[str] = []
        node = self.root
        while node.children:
            node = max(node.children.values(), key=lambda child: child.visits)
            if node.action is None:
                break
            moves.append(node.action.usi)
        return tuple(moves)

    @staticmethod
    def _check_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SearchCancelled("Search was cancelled")

    @staticmethod
    def _deadline_reached(deadline: float | None) -> bool:
        return deadline is not None and time.monotonic() >= deadline
