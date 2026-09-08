"""JSON persistence for cash, open/closed positions, and a fill ledger."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Position


@dataclass
class State:
    cash_usd: float = 0.0
    positions: List[Position] = field(default_factory=list)
    closed: List[Position] = field(default_factory=list)
    ledger: List[Dict] = field(default_factory=list)
    equity_history: List[Dict] = field(default_factory=list)  # [{"t": iso, "equity": x}]
    updated_at: Optional[str] = None

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.status == "open"]

    def equity(self) -> float:
        return self.cash_usd + sum(p.value_usd for p in self.open_positions)

    def to_dict(self) -> Dict:
        return {
            "cash_usd": self.cash_usd,
            "positions": [p.to_dict() for p in self.positions],
            "closed": [p.to_dict() for p in self.closed],
            "ledger": self.ledger[-2000:],
            "equity_history": self.equity_history[-5000:],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "State":
        return cls(
            cash_usd=float(d.get("cash_usd", 0.0)),
            positions=[Position.from_dict(x) for x in d.get("positions", [])],
            closed=[Position.from_dict(x) for x in d.get("closed", [])],
            ledger=list(d.get("ledger", [])),
            equity_history=list(d.get("equity_history", [])),
            updated_at=d.get("updated_at"),
        )


def load_state(path: str, starting_cash: float) -> State:
    if not os.path.exists(path):
        return State(cash_usd=starting_cash)
    with open(path, "r", encoding="utf-8") as fh:
        return State.from_dict(json.load(fh))


def save_state(path: str, state: State) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".lp_bot_state.", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
