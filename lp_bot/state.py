"""JSON persistence for cash, open/closed positions, and a fill ledger.

Saves are atomic (temp file + fsync + replace), keep the previous file as ``.bak``, and retry
when Windows briefly refuses the rename because another program has the file open. A run
holds an exclusive lock on ``<state>.lock`` so two copies of the bot cannot overwrite each
other's trades.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
            "ledger": self.ledger[-5000:],
            "equity_history": self.equity_history[-10000:],
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


def _read(path: str, attempts: int = 20, wait_s: float = 0.25) -> State:
    """Read state, retrying while another program briefly holds the file (Windows)."""
    for i in range(attempts):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            break
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(wait_s)
    return State.from_dict(json.loads(text))


def load_state(path: str, starting_cash: float, warn: Optional[Callable[[str], None]] = None) -> State:
    """Load state. A missing file starts a fresh paper account (deleting it is how you reset);
    a file that exists but is unreadable falls back to ``.bak``."""
    if not os.path.exists(path):
        if os.path.exists(path + ".bak") and warn:
            warn(f"{path} not found; starting a fresh paper account. An older copy exists at "
                 f"{path}.bak (delete it too if this is a deliberate reset)")
        return State(cash_usd=starting_cash)
    try:
        return _read(path)
    except OSError as exc:  # cannot open it at all: do not silently roll back to an older save
        raise RuntimeError(f"cannot open {path} ({exc}); close any program using it and retry") from exc
    except (ValueError, TypeError, KeyError) as exc:
        if os.path.exists(path + ".bak"):
            if warn:
                warn(f"{path} is unreadable ({exc}); loaded the previous save from {path}.bak")
            return _read(path + ".bak")
        raise RuntimeError(f"state file {path} is unreadable ({exc}) and there is no .bak copy") from exc


def _replace_with_retry(src: str, dst: str, attempts: int = 20, wait_s: float = 0.25) -> None:
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(wait_s)


def save_state(path: str, state: State) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".lp_bot_state.", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        if os.path.exists(path):
            try:
                shutil.copyfile(path, path + ".bak")
            except OSError:
                pass  # a missing backup must not block the save itself
        _replace_with_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class InstanceLock:
    """Exclusive, crash-safe lock on ``<state>.lock`` (released by the OS if the process dies)."""

    def __init__(self, state_path: str):
        self.path = os.path.abspath(state_path) + ".lock"
        self._fh = None

    def __enter__(self) -> "InstanceLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise RuntimeError(
                f"another lp_bot run is already using {self.path[:-5]}; stop it first "
                f"(status is safe to run alongside it)") from None
        self._fh = fh
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self._fh.close()
            self._fh = None
