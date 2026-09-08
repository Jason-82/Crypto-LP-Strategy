"""Position lifecycle: mark-to-model, exit rules, rotation, and sizing.

The manager never touches a chain directly; it asks the executor for fills. All rules are
driven by ``Config`` so they can be tuned without code changes. Fees are accrued from the
source's fee APY (an estimate; live adapters should replace ``fees_accrued_usd`` with
collected fees when they can read them on-chain).
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .config import Config
from .executors import Executor, tradable_live
from .il import impermanent_loss, in_range
from .models import Opportunity, Pool, Position
from .prices import VolEstimator
from .scoring import evaluate
from .state import State


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class Action:
    kind: str                      # "exit" | "enter" | "skip"
    reason: str
    position: Optional[Position] = None
    opportunity: Optional[Opportunity] = None
    size_usd: float = 0.0
    hard: bool = False


class Manager:
    def __init__(self, cfg: Config, state: State, executor: Executor, vol: VolEstimator):
        self.cfg = cfg
        self.state = state
        self.executor = executor
        self.vol = vol

    # ------------------------------------------------------------------ marking
    def mark(self, pos: Position, pool_now: Optional[Pool], ratio_now: Optional[float],
             now: datetime) -> None:
        """Update accrued fees, IL, and value for one open position."""
        last = parse_iso(pos.last_update or pos.opened_at)
        dt_years = max((now - last).total_seconds(), 0.0) / (365.0 * 86400.0)

        if ratio_now is not None and ratio_now > 0:
            pos.last_price_ratio = ratio_now
        k = pos.last_price_ratio / pos.entry_price_ratio if pos.entry_price_ratio > 0 else 1.0
        inr = in_range(k, pos.range_width)

        if pool_now is not None:
            fee_apr = self._fee_apr(pool_now)
            pos.last_apy = pool_now.apy
            pos.last_tvl = pool_now.tvl_usd
            pos.last_fee_apr = fee_apr
        else:
            fee_apr = pos.last_fee_apr or pos.entry_fee_apr

        if inr:
            pos.fees_accrued_usd += pos.size_usd * fee_apr * pos.fee_amp * dt_years
            pos.out_of_range_since = None
        elif pos.out_of_range_since is None:
            pos.out_of_range_since = iso(now)

        pos.il_usd = pos.size_usd * impermanent_loss(k, pos.range_width)
        pos.value_usd = pos.size_usd + pos.fees_accrued_usd + pos.il_usd
        pos.last_update = iso(now)

    @staticmethod
    def _fee_apr(pool: Pool) -> float:
        from .scoring import conservative_fee_apr
        return conservative_fee_apr(pool)[0]

    # -------------------------------------------------------------- exit rules
    def exit_reason(self, pos: Position, pool_now: Optional[Pool], now: datetime) -> Optional[Tuple[str, bool]]:
        """Return (reason, hard) if the position should be closed. Hard exits ignore
        the minimum hold time; soft exits wait for it so entry costs are earned back."""
        cfg = self.cfg
        held_h = (now - parse_iso(pos.opened_at)).total_seconds() / 3600.0

        if os.path.exists(cfg.kill_switch_path):
            return ("kill switch present", True)
        if pool_now is None:
            return ("pool no longer reported by source", True)
        if cfg.reject_outliers and pool_now.outlier:
            return ("source flagged pool as outlier", True)
        if pos.net_pnl_usd < -cfg.il_stop_pct * pos.size_usd:
            return (f"IL budget breached (net {pos.net_pnl_usd:+.2f} USD)", True)
        k = pos.last_price_ratio / pos.entry_price_ratio if pos.entry_price_ratio > 0 else 1.0
        if cfg.price_move_stop > 0 and k > 0 and abs(math.log(k)) > cfg.price_move_stop:
            return (f"price moved {k:.3f}x beyond stop", True)
        if pos.out_of_range_since is not None:
            oor_h = (now - parse_iso(pos.out_of_range_since)).total_seconds() / 3600.0
            if oor_h >= cfg.out_of_range_grace_hours:
                return (f"out of range for {oor_h:.0f}h", True)

        soft_ok = held_h >= cfg.min_hold_hours
        if soft_ok:
            floor_apy = max(cfg.min_apy_pct, pos.entry_apy * cfg.apy_decay_ratio)
            if pool_now.apy < floor_apy:
                return (f"yield decayed to {pool_now.apy:.1f}% (floor {floor_apy:.1f}%)", False)
            if pos.entry_tvl > 0 and pool_now.tvl_usd < pos.entry_tvl * cfg.tvl_drop_ratio:
                return (f"TVL fell to {pool_now.tvl_usd/1e6:.1f}M from {pos.entry_tvl/1e6:.1f}M", False)
            if held_h >= cfg.max_hold_days * 24.0:
                return (f"max hold of {cfg.max_hold_days}d reached; re-evaluate", False)
        return None

    # ---------------------------------------------------------------- planning
    def plan(self, opps: List[Opportunity], pools_by_id: Dict[str, Pool], now: datetime) -> List[Action]:
        cfg = self.cfg
        actions: List[Action] = []
        open_pos = self.state.open_positions
        held_ids = {p.pool_id for p in open_pos}

        # 1. exits
        exiting: set = set()
        for pos in open_pos:
            pool_now = pools_by_id.get(pos.pool_id)
            if pool_now is not None:
                pos.last_net_apr = evaluate(pool_now, cfg, self.vol).net_apr
            res = self.exit_reason(pos, pool_now, now)
            if res:
                actions.append(Action("exit", res[0], position=pos, hard=res[1]))
                exiting.add(pos.pool_id)

        if os.path.exists(cfg.kill_switch_path):
            return actions

        # 2. rotations: swap a stale position for a clearly better one
        cooldown = timedelta(hours=cfg.reentry_cooldown_hours)
        recently_closed = {p.pool_id for p in self.state.closed
                           if p.closed_at and now - parse_iso(p.closed_at) < cooldown}
        candidates = [o for o in opps if o.pool.pool_id not in held_ids
                      and o.pool.pool_id not in recently_closed]
        if cfg.mode == "live":
            candidates = [o for o in candidates if tradable_live(o.pool.project)]
        for pos in open_pos:
            if pos.pool_id in exiting or not candidates:
                continue
            held_h = (now - parse_iso(pos.opened_at)).total_seconds() / 3600.0
            if held_h < cfg.min_hold_hours:
                continue
            best = candidates[0]
            gain = best.net_apr - pos.last_net_apr
            if gain < cfg.rotation_margin_apr:
                continue
            horizon_y = cfg.horizon_days / 365.0
            switch_cost = cfg.gas_for(pos.chain) + cfg.gas_for(best.pool.chain) + \
                4.0 * pos.value_usd * cfg.slippage_bps_per_side / 10_000.0
            if gain * horizon_y * pos.value_usd > 2.0 * switch_cost:
                actions.append(Action("exit", f"rotate: {best.pool.symbol} on {best.pool.chain} "
                                      f"nets {gain*100:+.1f} pp", position=pos, hard=False))
                exiting.add(pos.pool_id)

        # 3. entries
        if self._daily_loss_breached(now):
            actions.append(Action("skip", "daily loss limit hit; no new entries"))
            return actions

        equity = self.state.equity()
        cash_after_exits = self.state.cash_usd
        for a in actions:
            if a.kind == "exit" and a.position:
                v = max(a.position.value_usd, 0.0)
                cash_after_exits += v * (1.0 - cfg.slippage_bps_per_side / 10_000.0) - cfg.gas_for(a.position.chain) / 2.0
        cash_avail = cash_after_exits - cfg.cash_reserve_pct * equity
        remaining = [p for p in open_pos if p.pool_id not in exiting]
        n_open = len(remaining)
        by_project: Dict[str, float] = {}
        by_chain: Dict[str, float] = {}
        for p in remaining:
            by_project[p.project] = by_project.get(p.project, 0.0) + p.value_usd
            by_chain[p.chain.lower()] = by_chain.get(p.chain.lower(), 0.0) + p.value_usd

        for o in candidates:
            if n_open >= cfg.max_positions or cash_avail < cfg.min_position_usd:
                break
            pool = o.pool
            cap = min(cfg.max_position_usd, cfg.max_pct_per_pool * equity)
            cap = min(cap, cfg.max_pct_per_project * equity - by_project.get(pool.project, 0.0))
            cap = min(cap, cfg.max_pct_per_chain * equity - by_chain.get(pool.chain.lower(), 0.0))
            size = min(cap, cash_avail)
            if size < cfg.min_position_usd:
                continue
            if size < cfg.max_position_usd:
                # costs are a larger share of a smaller position: re-check at the real size
                o = evaluate(pool, cfg, self.vol, size_usd=size)
                if o.net_apr < cfg.min_net_apr:
                    continue
            actions.append(Action("enter", f"score {o.score*100:.1f} / net {o.net_apr*100:.1f}%",
                                  opportunity=o, size_usd=size))
            cash_avail -= size
            n_open += 1
            by_project[pool.project] = by_project.get(pool.project, 0.0) + size
            by_chain[pool.chain.lower()] = by_chain.get(pool.chain.lower(), 0.0) + size
        return actions

    def _daily_loss_breached(self, now: datetime) -> bool:
        hist = self.state.equity_history
        if not hist:
            return False
        cutoff = now - timedelta(hours=24)
        ref = None
        for rec in hist:
            if parse_iso(rec["t"]) <= cutoff:
                ref = rec["equity"]
        if ref is None:
            ref = hist[0]["equity"]
        eq = self.state.equity()
        return ref > 0 and (eq - ref) / ref < -self.cfg.daily_loss_limit_pct

    # --------------------------------------------------------------- execution
    def execute(self, actions: List[Action], now: datetime) -> List[Dict]:
        log: List[Dict] = []
        for a in actions:
            if a.kind == "exit" and a.position is not None:
                fill = self.executor.close(a.position)
                pos = a.position
                if not fill.ok:
                    log.append(self._entry(now, "exit-failed", pos.pool_id, pos.symbol, 0.0, fill.note))
                    continue
                pos.status = "closed"
                pos.closed_at = iso(now)
                pos.close_reason = a.reason
                pos.exit_cost_usd = fill.cost_usd
                pos.realised_pnl_usd = fill.size_usd - (pos.size_usd + pos.entry_cost_usd)
                self.state.cash_usd = self.executor.cash_usd()
                self.state.positions = [p for p in self.state.positions if p.position_id != pos.position_id]
                self.state.closed.append(pos)
                log.append(self._entry(now, "exit", pos.pool_id, pos.symbol, fill.size_usd,
                                       f"{a.reason}; pnl {pos.realised_pnl_usd:+.2f}", fill.tx))
            elif a.kind == "enter" and a.opportunity is not None:
                o = a.opportunity
                fill = self.executor.open(o, a.size_usd)
                if not fill.ok:
                    log.append(self._entry(now, "enter-failed", o.pool.pool_id, o.pool.symbol, 0.0, fill.note))
                    continue
                _sig, _src, ratio = self.vol.estimate(o.pool)
                live_ratio = self.vol.current_ratio(o.pool)
                pos = Position(
                    position_id=uuid.uuid4().hex[:12], pool_id=o.pool.pool_id, chain=o.pool.chain,
                    project=o.pool.project, symbol=o.pool.symbol, opened_at=iso(now),
                    size_usd=fill.size_usd, entry_apy=o.pool.apy, entry_fee_apr=o.fee_apr,
                    entry_tvl=o.pool.tvl_usd, entry_price_ratio=live_ratio or ratio or 1.0,
                    range_width=o.range_width, amp=o.amp, fee_amp=o.fee_amp, sigma=o.sigma,
                    value_usd=fill.size_usd, last_price_ratio=live_ratio or ratio or 1.0,
                    last_apy=o.pool.apy, last_fee_apr=o.fee_apr, last_tvl=o.pool.tvl_usd,
                    last_net_apr=o.net_apr, last_update=iso(now), entry_cost_usd=fill.cost_usd,
                )
                self.state.positions.append(pos)
                self.state.cash_usd = self.executor.cash_usd()
                log.append(self._entry(now, "enter", o.pool.pool_id, o.pool.symbol, fill.size_usd,
                                       a.reason, fill.tx))
            else:
                log.append(self._entry(now, a.kind, "", "", 0.0, a.reason))
        self.state.ledger.extend(log)
        return log

    @staticmethod
    def _entry(now: datetime, kind: str, pool_id: str, symbol: str, usd: float, note: str,
               tx: Optional[str] = None) -> Dict:
        return {"t": iso(now), "kind": kind, "pool_id": pool_id, "symbol": symbol,
                "usd": round(usd, 2), "note": note, "tx": tx}

    # -------------------------------------------------------------------- tick
    def tick(self, pools: List[Pool], opps: List[Opportunity], now: Optional[datetime] = None,
             ratios: Optional[Dict[str, float]] = None) -> Dict:
        """One full cycle: mark positions, plan, execute, record equity."""
        now = now or utcnow()
        pools_by_id = {p.pool_id: p for p in pools}
        for pos in self.state.open_positions:
            pool_now = pools_by_id.get(pos.pool_id)
            if ratios is not None and pos.pool_id in ratios:
                ratio_now = ratios[pos.pool_id]
            else:
                ratio_now = self.vol.current_ratio(pool_now) if pool_now else None
            self.mark(pos, pool_now, ratio_now, now)
        actions = self.plan(opps, pools_by_id, now)
        log = self.execute(actions, now)
        self.state.updated_at = iso(now)
        self.state.equity_history.append({"t": iso(now), "equity": round(self.state.equity(), 2)})
        return {"now": iso(now), "actions": log, "equity": self.state.equity(),
                "cash": self.state.cash_usd, "open": len(self.state.open_positions)}

