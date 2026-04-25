"""Adaptive curriculum scheduler (Section 4.4 of the plan).

Maintains a moving-average logical-correction rate per level and promotes
the agent to harder levels once the threshold is met. Implements the
Section 4.4 mixing rules:

* Stay at L1 until L1 hits 80%.
* Then mix L1/L2 with weights 30/70 until L2 hits 70%.
* Then unlock L3 at 30% weight (with L1/L2 sharing the remaining 70%).

The scheduler is *override-able* - eval scripts pass ``forced_level`` to
hold one configuration steady.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from qubit_medic.config import CURRICULUM, CurriculumLevel, level_by_name


# --------------------------------------------------------------------------- #
# Per-level moving average                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class _MovingWindow:
    window_size: int = 100
    history: deque[float] = field(default_factory=deque)

    def push(self, value: float) -> None:
        self.history.append(value)
        while len(self.history) > self.window_size:
            self.history.popleft()

    def mean(self) -> float:
        return sum(self.history) / len(self.history) if self.history else 0.0

    def __len__(self) -> int:
        return len(self.history)


# --------------------------------------------------------------------------- #
# Scheduler                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class CurriculumScheduler:
    """Picks a curriculum level for each new episode."""

    rng: random.Random = field(default_factory=lambda: random.Random(42))
    windows: dict[str, _MovingWindow] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for lvl in CURRICULUM:
            self.windows.setdefault(lvl.name, _MovingWindow())

    # ----- public API -----------------------------------------------------

    def update(self, level_name: str, logical_correction: float) -> None:
        """Record one episode's logical-correction outcome."""
        self.windows[level_name].push(float(logical_correction))

    def sample(self, forced_level: Optional[str] = None) -> CurriculumLevel:
        """Return the level to use for the next episode."""
        if forced_level is not None:
            return level_by_name(forced_level)

        l1, l2, l3 = (level_by_name(n) for n in ("L1_warmup", "L2_target", "L3_stretch"))
        l1_rate = self.windows["L1_warmup"].mean()
        l2_rate = self.windows["L2_target"].mean()
        l1_n = len(self.windows["L1_warmup"])
        l2_n = len(self.windows["L2_target"])

        # Phase A: still working on L1.
        if l1_n < 30 or l1_rate < l1.promotion_threshold:
            return l1

        # Phase B: L1 unlocked, mixing L1 (30%) and L2 (70%).
        if l2_n < 30 or l2_rate < l2.promotion_threshold:
            return l1 if self.rng.random() < 0.30 else l2

        # Phase C: L3 unlocked, splits 20% L1, 50% L2, 30% L3.
        roll = self.rng.random()
        if roll < 0.20:
            return l1
        if roll < 0.70:
            return l2
        return l3

    # ----- introspection (used by /state endpoint and logs) ---------------

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "moving_mean": w.mean(),
                "samples": float(len(w)),
            }
            for name, w in self.windows.items()
        }
