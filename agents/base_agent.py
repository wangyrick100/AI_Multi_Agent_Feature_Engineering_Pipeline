"""
Base agent class: provides logging, timing, and the standard run() contract
that every derived agent must implement.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.data_models import PipelineState
    from core.config import Config


class BaseAgent(ABC):
    """
    Abstract base for every agent in the Feature Engineering System.

    Subclasses implement `_run(df, state)` and may optionally override
    `validate_inputs()` for pre-flight checks.
    """

    name: str = "BaseAgent"

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.logger = logging.getLogger(self.name)

    # ── Public interface ────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, state: "PipelineState") -> "PipelineState":
        """
        Execute this agent, update *state* in-place, and return it.
        Wraps _run with timing, logging, and error capture.
        """
        self.logger.info("▶ %s starting …", self.name)
        t0 = time.perf_counter()
        try:
            self.validate_inputs(df, state)
            state = self._run(df, state)
            elapsed = time.perf_counter() - t0
            state.agent_timings[self.name] = round(elapsed, 3)
            state.touch(self.name)
            self.logger.info("✔ %s completed in %.2fs", self.name, elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = f"{self.name} failed after {elapsed:.2f}s: {exc}"
            self.logger.error(msg, exc_info=True)
            state.errors.append(msg)
            state.status = f"error:{self.name}"
        return state

    # ── Subclass contract ───────────────────────────────────────────────────

    @abstractmethod
    def _run(self, df: pd.DataFrame, state: "PipelineState") -> "PipelineState":
        """Core logic; must return the (possibly mutated) state."""

    def validate_inputs(self, df: pd.DataFrame, state: "PipelineState") -> None:
        """Override to add pre-flight validation; raise ValueError on failure."""
        if df is None or df.empty:
            raise ValueError(f"{self.name}: received empty DataFrame.")
