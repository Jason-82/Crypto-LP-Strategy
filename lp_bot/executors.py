"""Execution layer.

``PaperExecutor`` simulates fills with slippage and gas so the whole strategy can run
end-to-end without a wallet. ``LiveExecutor`` is the boundary to real money: it resolves a
per-protocol adapter and refuses to run until one is registered and the operator has
explicitly acknowledged live mode. No adapter ships with this repository; see README.
"""

from __future__ import annotations

import os
import uuid
from typing import Callable, Dict, Optional

from .config import Config
from .models import Fill, Opportunity, Position

LIVE_ACK_ENV = "LP_BOT_LIVE_ACK"
LIVE_ACK_VALUE = "I_UNDERSTAND_THIS_TRADES_REAL_FUNDS"
PRIVATE_KEY_ENV = "LP_BOT_PRIVATE_KEY"  # never written to disk or logs by this package


class Executor:
    name = "base"

    def cash_usd(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def open(self, opp: Opportunity, size_usd: float) -> Fill:  # pragma: no cover
        raise NotImplementedError

    def close(self, pos: Position) -> Fill:  # pragma: no cover
        raise NotImplementedError


class PaperExecutor(Executor):
    name = "paper"

    def __init__(self, cfg: Config, cash_usd: float):
        self.cfg = cfg
        self._cash = float(cash_usd)

    def cash_usd(self) -> float:
        return self._cash

    def _cost(self, chain: str, size_usd: float) -> float:
        return self.cfg.gas_for(chain) / 2.0 + size_usd * self.cfg.slippage_bps_per_side / 10_000.0

    def open(self, opp: Opportunity, size_usd: float) -> Fill:
        if size_usd <= 0 or size_usd > self._cash:
            return Fill(ok=False, note="insufficient cash")
        cost = self._cost(opp.pool.chain, size_usd)
        placed = size_usd - cost
        if placed <= 0:
            return Fill(ok=False, note="size smaller than entry cost")
        self._cash -= size_usd
        return Fill(ok=True, size_usd=placed, cost_usd=cost, tx=f"paper-{uuid.uuid4().hex[:10]}",
                    note="simulated mint")

    def close(self, pos: Position) -> Fill:
        gross = max(pos.value_usd, 0.0)
        cost = self._cost(pos.chain, gross)
        returned = max(gross - cost, 0.0)
        self._cash += returned
        return Fill(ok=True, size_usd=returned, cost_usd=cost, tx=f"paper-{uuid.uuid4().hex[:10]}",
                    note="simulated burn")


AdapterFactory = Callable[[Config], "ProtocolAdapter"]
_ADAPTERS: Dict[str, AdapterFactory] = {}


class ProtocolAdapter:
    """What a live adapter must provide for one protocol family (e.g. uniswap-v3 on EVM).

    Implementations are expected to:
      * derive the wallet from LP_BOT_PRIVATE_KEY (or a hardware/remote signer) at
        construction time and never log or persist it;
      * ``open``: acquire both legs (swap from the funding asset), approve, mint the LP
        position (for concentrated pools: a symmetric range of ``opp.range_width``), and
        return the capital actually placed after gas and slippage;
      * ``close``: burn/withdraw, collect fees, swap back to the funding asset, and
        return the USD realised;
      * enforce on-chain slippage limits equal to ``cfg.slippage_bps_per_side``;
      * be validated on a testnet or with dust amounts before being trusted.
    """

    def cash_usd(self) -> float:  # pragma: no cover
        raise NotImplementedError

    def open(self, opp: Opportunity, size_usd: float) -> Fill:  # pragma: no cover
        raise NotImplementedError

    def close(self, pos: Position) -> Fill:  # pragma: no cover
        raise NotImplementedError


def register_adapter(project_prefix: str, factory: AdapterFactory) -> None:
    """Register a live adapter for DefiLlama project slugs starting with ``project_prefix``."""
    _ADAPTERS[project_prefix.lower()] = factory


def adapter_for(project: str) -> Optional[str]:
    slug = project.lower()
    matches = [k for k in _ADAPTERS if slug.startswith(k)]
    return max(matches, key=len) if matches else None


class LiveExecutor(Executor):
    name = "live"

    def __init__(self, cfg: Config):
        if os.environ.get(LIVE_ACK_ENV) != LIVE_ACK_VALUE:
            raise RuntimeError(
                f"live mode requires {LIVE_ACK_ENV}={LIVE_ACK_VALUE} in the environment")
        if not os.environ.get(PRIVATE_KEY_ENV):
            raise RuntimeError(f"live mode requires {PRIVATE_KEY_ENV} (never put it in a file in this repo)")
        if not _ADAPTERS:
            raise RuntimeError(
                "no live protocol adapters are registered; implement ProtocolAdapter for the "
                "protocols you want to trade and call register_adapter() before starting")
        self.cfg = cfg
        self._instances: Dict[str, ProtocolAdapter] = {}

    def _adapter(self, project: str) -> ProtocolAdapter:
        key = adapter_for(project)
        if key is None:
            raise NotImplementedError(f"no live adapter registered for project '{project}'")
        if key not in self._instances:
            self._instances[key] = _ADAPTERS[key](self.cfg)
        return self._instances[key]

    def cash_usd(self) -> float:
        # Funding balance is adapter-defined (usually a stablecoin on the funding chain).
        # Use the first registered adapter as the treasury view.
        first = next(iter(_ADAPTERS))
        return self._adapter(first).cash_usd()

    def open(self, opp: Opportunity, size_usd: float) -> Fill:
        return self._adapter(opp.pool.project).open(opp, size_usd)

    def close(self, pos: Position) -> Fill:
        return self._adapter(pos.project).close(pos)


def tradable_live(project: str) -> bool:
    return adapter_for(project) is not None
