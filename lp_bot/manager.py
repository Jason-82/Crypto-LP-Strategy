"""Position lifecycle: mark-to-market, exit rules, rotation, and sizing.

Valuation is in USD. Each position records the USD prices of its two legs at entry, so every
mark splits the result into what simply holding the tokens would have made (``price_pnl``),
the LP's shortfall versus holding (``il``), fees, and rewards:

    value = size * position_value(k) * qb + fees + rewards
    k  = (Pa/Pb now) / (Pa/Pb at entry)          qb = Pb now / Pb at entry

Exit rules come in two kinds. Hard exits (kill switch, IL budget, de-peg, out of range, pool
gone, unpriceable) fire immediately. Soft exits wait ``min_hold_hours`` and are judged on the
position's forward economics: smoothed fees, expected IL, and only the exit cost (the entry
cost is already spent). Rotation swaps at most ``max_rotations_per_cycle`` weak positions for
candidates that free cash could not fund, one position per candidate.

The manager never touches a chain; it asks the executor for fills.
"""

from __future__ import annotations

import dataclasses
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .config import Config
from .executors import Executor, tradable_live
from .il import amplification, expected_il_rate, hold_value, impermanent_loss, in_range, position_value
from .models import Opportunity, Pool, Position
from .prices import VolEstimator, pair_key, peg_kind
from .scoring import evaluate, marking_fee_apr, reject_reasons

CANARY_KEY = "coingecko:usd-coin"   # always requested, so feed health never depends on holdings
from .state import State


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def hours_between(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


@dataclass
class Action:
    kind: str                      # "exit" | "enter" | "skip"
    reason: str
    position: Optional[Position] = None
    opportunity: Optional[Opportunity] = None
    size_usd: float = 0.0
    hard: bool = False
    prices: Optional[List[float]] = None
    legs: Optional[Tuple[str, str, str, str]] = None
    peg: Optional[List[int]] = None


@dataclass
class CycleContext:
    feed_up: bool = True        # the yields feed returned a plausible universe this cycle
    price_feed_ok: bool = True  # the price source answered at all this cycle
    priced: int = 0             # open positions priced this cycle
    unpriced: int = 0
    migrated: int = 0
    migration_delta: float = 0.0  # equity change caused purely by revaluing old positions

    @property
    def prices_up(self) -> bool:
        return self.price_feed_ok


class Manager:
    def __init__(self, cfg: Config, state: State, executor: Executor, vol: VolEstimator):
        self.cfg = cfg
        self.state = state
        self.executor = executor
        self.vol = vol
        self.ctx = CycleContext()
        self._cycle_keys: Dict[str, List[str]] = {}   # keys that priced each position this cycle
        self.notes: List[str] = []
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ pricing
    @staticmethod
    def _synthetic_pool(pos: Position) -> Pool:
        return Pool(pos.pool_id, pos.chain, pos.project, pos.symbol, 0.0, 0.0)

    def _legs_for(self, pos: Position, pool_now: Optional[Pool]) -> Optional[List[str]]:
        if len(pos.leg_keys) == 2 and all(pos.leg_keys):
            return pos.leg_keys
        ka, kb, sa, sb = self.vol.legs(pool_now or self._synthetic_pool(pos))
        if not ka or not kb or ka == kb:
            return None
        if not pos.leg_symbols:
            pos.leg_symbols = [sa, sb]
        if pool_now is not None:   # address keys from the feed are kept; symbol guesses are not
            pos.leg_keys = [ka, kb]
        return [ka, kb]

    def _price_positions(self, positions: List[Position], pools_by_id: Dict[str, Pool]) -> Dict[str, Optional[List[float]]]:
        keys_by_pos = {p.position_id: self._legs_for(p, pools_by_id.get(p.pool_id)) for p in positions}
        self._cycle_keys = {pid: ks for pid, ks in keys_by_pos.items() if ks}
        wanted = sorted({k for ks in keys_by_pos.values() if ks for k in ks} | {CANARY_KEY})
        got = self.vol.current_prices(wanted)
        self.ctx.price_feed_ok = bool(got)
        out: Dict[str, Optional[List[float]]] = {}
        retry: List[Position] = []
        for pos in positions:
            ks = keys_by_pos[pos.position_id]
            if ks and all(got.get(k, 0) > 0 for k in ks):
                out[pos.position_id] = [got[ks[0]], got[ks[1]]]
            else:
                out[pos.position_id] = None
                retry.append(pos)
        if retry and self.ctx.price_feed_ok:
            # fall back from address keys to CoinGecko ids for the same two tokens
            alt: Dict[str, Tuple[str, str]] = {}
            for pos in retry:
                base = pools_by_id.get(pos.pool_id) or self._synthetic_pool(pos)
                for ka, kb, sa, sb in self.vol.leg_candidates(base):
                    if [ka, kb] != pos.leg_keys and (not pos.leg_symbols or [sa, sb] == pos.leg_symbols):
                        alt[pos.position_id] = (ka, kb)
                        break
            if alt:
                got2 = self.vol.current_prices(sorted({k for pair in alt.values() for k in pair}))
                for pos in retry:
                    pair = alt.get(pos.position_id)
                    if pair and got2.get(pair[0], 0) > 0 and got2.get(pair[1], 0) > 0:
                        # this cycle only: next cycle tries the address keys again
                        out[pos.position_id] = [got2[pair[0]], got2[pair[1]]]
                        self._cycle_keys[pos.position_id] = list(pair)
        return out

    def _entry_prices(self, pool: Pool) -> Tuple[Optional[Tuple[List[float], Tuple[str, str, str, str], List[int]]], str]:
        """Live USD prices for both legs, the legs used, and which legs are pegged. Refuses a
        pool whose stablecoin leg is already off its peg."""
        cands = self.vol.leg_candidates(pool)
        if not cands:
            return None, "cannot price both tokens"
        got = self.vol.current_prices(sorted({k for c in cands for k in c[:2]}))
        for ka, kb, sa, sb in cands:
            pa, pb = got.get(ka, 0), got.get(kb, 0)
            if pa > 0 and pb > 0:
                peg: List[int] = []
                for i, (sym, px) in enumerate(((sa, pa), (sb, pb))):
                    kind = peg_kind(sym)
                    if kind == "usd" and 0.5 <= px <= 2.0:
                        if abs(px - 1.0) > self.cfg.depeg_stop_pct:
                            return None, f"{sym} is off its peg at ${px:.4f}"
                        peg.append(i)
                    elif kind == "yield" and 0.9 <= px <= 3.0:
                        if px < 1.0 - self.cfg.depeg_stop_pct:
                            return None, f"{sym} trades below $1 at ${px:.4f}"
                        peg.append(i)
                return ([pa, pb], (ka, kb, sa, sb), peg), ""
        return None, "cannot price both tokens"

    @staticmethod
    def _k_qb(pos: Position) -> Tuple[float, float]:
        """Price-ratio change since entry, and the quote leg's USD change since entry."""
        if len(pos.entry_prices) == 2 and len(pos.last_prices) == 2 and \
                min(pos.entry_prices + pos.last_prices) > 0:
            pa0, pb0 = pos.entry_prices
            pa, pb = pos.last_prices
            return (pa / pb) / (pa0 / pb0), pb / pb0
        if pos.entry_price_ratio > 0 and pos.last_price_ratio > 0:
            return pos.last_price_ratio / pos.entry_price_ratio, 1.0
        return 1.0, 1.0

    def _migrate(self, pos: Position, prices: List[float], pool_now: Optional[Pool], now: datetime,
                 keys: Optional[List[str]] = None) -> None:
        """Give a position opened by an older version a USD price basis.

        The recorded entry ratio is kept so IL is continuous. The USD level comes from a
        stablecoin leg (worth $1 at entry), else from historical prices at the entry time,
        else from today's price (and the note says so). A stored 1.0 that was really a
        placeholder is replaced by the historical ratio when one is available."""
        pa, pb = prices
        r_now = pa / pb
        syms = pos.leg_symbols if len(pos.leg_symbols) == 2 else ["", ""]
        kinds = [peg_kind(s_) for s_ in syms]
        keys = keys if keys and len(keys) == 2 else pos.leg_keys
        hist: Dict[str, float] = {}
        if len(keys) == 2:
            hist = self.vol.historical_prices(keys, int(parse_iso(pos.opened_at).timestamp()))
        ha, hb = (hist.get(keys[0]), hist.get(keys[1])) if hist else (None, None)
        r_hist = ha / hb if ha and hb else None

        er = pos.entry_price_ratio
        if er <= 0 or abs(er - 1.0) < 1e-12:
            if r_hist:
                er = r_hist
                self.notes.append(f"{pos.symbol}: entry price was never recorded; recovered it from price history")
            elif kinds[0] == "usd" and kinds[1] == "usd":
                er = 1.0
            else:
                er = r_now
                self.notes.append(f"{pos.symbol}: entry price was never recorded; IL tracked from now")
            pos.entry_price_ratio = er

        real = True
        if kinds[1] == "usd":
            pos.entry_prices = [er * 1.0, 1.0]
        elif kinds[0] == "usd":
            pos.entry_prices = [1.0, 1.0 / er]
        elif hb:
            pos.entry_prices = [er * hb, hb]
        else:
            pos.entry_prices = [er * pb, pb]
            real = False
        pos.peg_legs = [i for i in (0, 1) if real and kinds[i] in ("usd", "yield")
                        and 0.5 <= pos.entry_prices[i] <= 3.0]
        pos.price_basis_at = iso(now)
        if pool_now is not None:
            o = evaluate(pool_now, self.cfg, self.vol, size_usd=max(pos.size_usd, self.cfg.min_position_usd), holding=True)
            pos.drift_apr = o.drift
            if (o.range_width or 0.0) != (pos.range_width or 0.0):
                pos.range_width, pos.amp, pos.fee_amp, pos.sigma = o.range_width, o.amp, o.fee_amp, o.sigma
                rng = "full range" if o.range_width is None else f"a +/-{(o.range_width - 1) * 100:.1f}% range"
                self.notes.append(f"{pos.symbol}: now modelled as {rng}")
        if real:
            self.notes.append(f"{pos.symbol}: switched to USD valuation (history since entry kept)")
        else:
            self.notes.append(f"{pos.symbol}: switched to USD valuation; {syms[1] or 'the quote token'}'s "
                              f"own USD move before today is not counted")
        self.ctx.migrated += 1

    # ------------------------------------------------------------------ marking
    def mark(self, pos: Position, pool_now: Optional[Pool], prices_now: Optional[List[float]],
             now: datetime) -> None:
        """Accrue fees and rewards since the last mark and revalue the position in USD."""
        cfg = self.cfg
        last = parse_iso(pos.last_update or pos.opened_at)
        dt_h = max(hours_between(last, now), 0.0)
        was_in_range = pos.out_of_range_since is None

        legacy_basis = None
        if prices_now:
            if not pos.entry_prices:
                legacy_basis = (pos.entry_price_ratio, pos.range_width, prices_now[0] / prices_now[1])
                keys = self._cycle_keys.get(pos.position_id) or pos.leg_keys
                self._migrate(pos, prices_now, pool_now, now, keys)
            pos.last_prices = list(prices_now)
            pos.last_price_ratio = prices_now[0] / prices_now[1]
            pos.price_stale_since = None
        elif pos.price_stale_since is None:
            pos.price_stale_since = iso(now)

        k, qb = self._k_qb(pos)
        inr = in_range(k, pos.range_width)

        if pool_now is not None:
            pos.missing_since = None
            fee_now = marking_fee_apr(pool_now)
            rew_now = (pool_now.apy_reward or 0.0) / 100.0 * cfg.reward_haircut
            pos.last_apy, pos.last_tvl = pool_now.apy, pool_now.tvl_usd
        elif self.ctx.feed_up:
            # the feed works but this pool is not in it: earn nothing until it reappears
            if pos.missing_since is None:
                pos.missing_since = iso(now)
            fee_now = rew_now = 0.0
        else:
            fee_now, rew_now = pos.last_fee_apr, pos.last_reward_apr

        if pos.entry_prices:
            lp_now = pos.size_usd * position_value(k, pos.range_width) * qb
        else:
            # never priced in USD yet: keep the old LP-versus-holding valuation, which is
            # unit-free, rather than reading a ratio in the quote token as dollars
            lp_now = pos.size_usd * (1.0 + impermanent_loss(k, pos.range_width))
        if dt_h > cfg.gap_hours:
            # The bot was not running. Book the gap at the lower of the before/after rates, only
            # if the position was in range at both ends, and not at all if today's rate is unknown.
            fee_rate, rew_rate = min(fee_now, pos.last_fee_apr), min(rew_now, pos.last_reward_apr)
            known = pool_now is not None
            accrue = inr and was_in_range and known
            pos.gap_hours += dt_h
            why = ("fees at " + format(fee_rate * 100, ".1f") + "%" if accrue
                   else "no fees: pool not in today's feed" if not known else "no fees: out of range")
            self.notes.append(f"{pos.symbol}: {dt_h:.0f}h unmarked gap booked conservatively ({why})")
        else:
            fee_rate, rew_rate, accrue = fee_now, rew_now, inr
        years = dt_h / 8760.0
        if accrue:
            base = max(lp_now, 0.0)  # fees are earned on the liquidity's current value
            pos.fees_accrued_usd += base * fee_rate * pos.fee_amp * years
            pos.rewards_accrued_usd += base * rew_rate * years
        if pool_now is not None:
            pos.last_fee_apr, pos.last_reward_apr = fee_now, rew_now

        if inr:
            pos.out_of_range_since = None
        elif pos.out_of_range_since is None:
            pos.out_of_range_since = iso(now)

        hold = pos.size_usd * hold_value(k) * qb if pos.entry_prices else pos.size_usd
        pos.il_usd = lp_now - hold
        pos.price_pnl_usd = hold - pos.size_usd
        pos.value_usd = lp_now + pos.fees_accrued_usd + pos.rewards_accrued_usd
        pos.last_update = iso(now)
        if legacy_basis is not None:
            er_old, rw_old, r_now = legacy_basis
            k_old = r_now / er_old if er_old > 0 else 1.0
            old_value = pos.size_usd * (1.0 + impermanent_loss(k_old, rw_old)) \
                + pos.fees_accrued_usd + pos.rewards_accrued_usd
            self.ctx.migration_delta += pos.value_usd - old_value

    def _mark_all(self, pools_by_id: Dict[str, Pool], now: datetime) -> None:
        positions = self.state.open_positions
        priced = self._price_positions(positions, pools_by_id)
        self.ctx.priced = sum(1 for v in priced.values() if v)
        self.ctx.unpriced = len(priced) - self.ctx.priced
        self.ctx.migrated, self.ctx.migration_delta = 0, 0.0
        for pos in positions:
            self.mark(pos, pools_by_id.get(pos.pool_id), priced.get(pos.position_id), now)
        if self.ctx.migration_delta:
            # A one-off revaluation is not a trading result: shift the recent equity points by
            # the same amount so the daily loss limit compares like with like.
            for rec in self.state.equity_history:
                if hours_between(parse_iso(rec["t"]), now) <= 36.0:
                    rec["equity"] = round(rec["equity"] + self.ctx.migration_delta, 2)
        if not self.ctx.price_feed_ok and positions:
            self.warnings.append("price source returned nothing this cycle; holding, no new entries")
        elif self.ctx.unpriced:
            names = ", ".join(p.symbol for p in positions if not priced.get(p.position_id))
            self.warnings.append(f"could not price {self.ctx.unpriced} position(s) this cycle: {names}")

    # -------------------------------------------------------------- exit rules
    def _peg_breach(self, pos: Position, now: datetime) -> Optional[str]:
        stop = self.cfg.depeg_stop_pct
        for i in pos.peg_legs:
            if i >= len(pos.last_prices) or i >= len(pos.entry_prices):
                continue
            px, ref = pos.last_prices[i], pos.entry_prices[i]
            if px <= 0 or ref <= 0:
                continue
            sym = pos.leg_symbols[i] if i < len(pos.leg_symbols) else f"leg {i}"
            if peg_kind(sym) == "yield":  # accrues above $1 by design: only a fall counts
                ref = max(ref, 1.0)
                if px < ref * (1.0 - stop):
                    return f"{sym} fell {(1 - px / ref) * 100:.1f}% below its entry value"
            elif abs(px - 1.0) > stop:
                return f"{sym} off its $1 peg at ${px:.4f}"
        if pos.sigma < self.cfg.peg_pair_sigma and pos.price_basis_at:
            k, _ = self._k_qb(pos)
            held_y = max(hours_between(parse_iso(pos.opened_at), now), 0.0) / 8760.0
            dev = math.log(k) - pos.drift_apr * held_y if k > 0 else 0.0
            if abs(dev) > stop:
                return f"pegged pair diverged {dev * 100:+.1f}% from its expected ratio"
        return None

    def _hold_view(self, pos: Position, pool_now: Pool) -> Opportunity:
        """Forward economics of keeping this position, using its own range model."""
        h = evaluate(pool_now, self.cfg, self.vol, size_usd=max(pos.value_usd, self.cfg.min_position_usd), holding=True)
        amp = pos.amp if pos.amp > 0 else amplification(pos.range_width)
        if abs(amp - h.amp) > 1e-9 or abs(pos.fee_amp - h.fee_amp) > 1e-9:
            il = expected_il_rate(h.sigma, amp)
            net = h.net_apr + h.expected_il_apr - il + h.fee_apr * (pos.fee_amp - h.fee_amp)
            h = dataclasses.replace(h, net_apr=net, expected_il_apr=il, amp=amp, fee_amp=pos.fee_amp,
                                    range_width=pos.range_width, score=net * h.confidence)
        return h

    def exit_reason(self, pos: Position, pool_now: Optional[Pool], now: datetime,
                    hold: Optional[Opportunity] = None, hard_only: bool = False) -> Optional[Tuple[str, bool]]:
        """Return (reason, hard) if the position should be closed."""
        cfg = self.cfg
        if os.path.exists(cfg.kill_switch_path):
            return ("kill switch present", True)
        if pool_now is not None:
            blocked = [r for r in reject_reasons(pool_now, cfg) if r.endswith("blocked")]
            if blocked:
                return (f"pool now excluded by config ({blocked[0]})", True)
        if pos.missing_since and self.ctx.feed_up:
            mh = hours_between(parse_iso(pos.missing_since), now)
            if mh >= cfg.missing_grace_hours:
                return (f"pool missing from the feed for {mh:.0f}h", True)
        if pos.price_stale_since and self.ctx.prices_up and self.ctx.feed_up and pool_now is not None:
            # the pool is live and prices work, yet this position cannot be priced
            sh = hours_between(parse_iso(pos.price_stale_since), now)
            if sh >= cfg.price_stale_hours:
                return (f"could not price the position for {sh:.0f}h", True)
        if pos.net_pnl_usd < -cfg.il_stop_pct * pos.size_usd:
            return (f"IL budget breached (fees+IL {pos.net_pnl_usd:+.2f} USD)", True)
        peg = self._peg_breach(pos, now)
        if peg:
            return (peg, True)
        k, qb = self._k_qb(pos)
        if cfg.price_move_stop > 0 and k > 0 and abs(math.log(k)) > cfg.price_move_stop:
            return (f"price ratio moved {k:.3f}x beyond stop", True)
        if pos.out_of_range_since is not None:
            oor_h = hours_between(parse_iso(pos.out_of_range_since), now)
            if oor_h >= cfg.out_of_range_grace_hours:
                return (f"out of range for {oor_h:.0f}h", True)

        held_h = hours_between(parse_iso(pos.opened_at), now)
        if hard_only or pool_now is None or held_h < cfg.min_hold_hours or pos.price_stale_since:
            return None  # never sell on soft grounds a position we could not price this cycle
        hold = hold or self._hold_view(pos, pool_now)
        if hold.net_apr < cfg.min_hold_apr:
            return (f"expected to lose money from here: fees and rewards {hold.fee_apr * 100:.1f}%+"
                    f"{hold.reward_apr * 100:.1f}% vs expected IL {hold.expected_il_apr * 100:.1f}%", False)
        if cfg.apy_decay_ratio > 0 and pos.entry_fee_apr > 0 and hold.fee_apr < pos.entry_fee_apr * cfg.apy_decay_ratio:
            fresh = evaluate(pool_now, cfg, self.vol, size_usd=max(pos.value_usd, cfg.min_position_usd))
            if fresh.net_apr < cfg.min_net_apr:
                return (f"fee rate fell to {hold.fee_apr * 100:.1f}% from the {pos.entry_fee_apr * 100:.1f}% "
                        f"bought on, and the pool would not be bought today", False)
        if pos.entry_tvl > 0 and cfg.tvl_drop_ratio > 0:
            adjusted = pool_now.tvl_usd / max(hold_value(k) * qb, 1e-9)  # strip market moves
            if adjusted < pos.entry_tvl * cfg.tvl_drop_ratio:
                return (f"liquidity left the pool: TVL {pool_now.tvl_usd / 1e6:.1f}M vs "
                        f"{pos.entry_tvl / 1e6:.1f}M at entry, after price moves", False)
        if held_h >= cfg.max_hold_days * 24.0:
            fresh = evaluate(pool_now, cfg, self.vol, size_usd=max(pos.value_usd, cfg.min_position_usd))
            if fresh.net_apr < cfg.min_net_apr:
                return (f"held {cfg.max_hold_days}d and would not be bought today", False)
        return None

    # ---------------------------------------------------------------- planning
    def _proceeds(self, pos: Position) -> float:
        v = max(pos.value_usd, 0.0)
        stable = pos.sigma < self.cfg.stable_sigma_threshold
        return max(v - self.cfg.one_way_cost(pos.chain, v, stable), 0.0)

    def _cap(self, pool: Pool, equity: float, by_project: Dict[str, float], by_chain: Dict[str, float]) -> float:
        cfg = self.cfg
        cap = min(cfg.max_position_usd, cfg.max_pct_per_pool * equity, cfg.max_share_of_pool_tvl * pool.tvl_usd)
        cap = min(cap, cfg.max_pct_per_project * equity - by_project.get(pool.project, 0.0))
        return min(cap, cfg.max_pct_per_chain * equity - by_chain.get(pool.chain.lower(), 0.0))

    def _daily_loss_breached(self, now: datetime) -> bool:
        """Equity versus a reference about a day old; after a restart, the first point after it.
        One-off revaluations are netted out of the history when they happen."""
        hist = self.state.equity_history
        if not hist:
            return False
        pts = [(parse_iso(r["t"]), r) for r in hist]
        window = [(t, r) for t, r in pts if hours_between(t, now) <= 36.0]
        if not window:
            return False
        older = [(t, r) for t, r in window if hours_between(t, now) >= 24.0]
        ref = older[-1][1] if older else window[0][1]
        eq = self.state.equity()
        ref_eq = ref["equity"]
        return ref_eq > 0 and (eq - ref_eq) / ref_eq < -self.cfg.daily_loss_limit_pct

    def plan(self, opps: List[Opportunity], pools_by_id: Dict[str, Pool], now: datetime,
             hard_only: bool = False) -> List[Action]:
        cfg = self.cfg
        actions: List[Action] = []
        open_pos = self.state.open_positions
        holds: Dict[str, Opportunity] = {}
        exiting: Dict[str, Position] = {}

        # 1. exits
        for pos in open_pos:
            pool_now = pools_by_id.get(pos.pool_id)
            if pool_now is not None and not hard_only:
                h = self._hold_view(pos, pool_now)
                holds[pos.position_id] = h
                pos.last_hold_apr = pos.last_net_apr = h.net_apr
            res = self.exit_reason(pos, pool_now, now, holds.get(pos.position_id), hard_only)
            if res:
                actions.append(Action("exit", res[0], position=pos, hard=res[1]))
                exiting[pos.position_id] = pos

        if hard_only or os.path.exists(cfg.kill_switch_path):
            return actions
        if not self.ctx.feed_up:
            actions.append(Action("skip", "yields feed unavailable; holding, no new entries"))
            return actions
        if not self.ctx.prices_up:
            actions.append(Action("skip", "price feed unavailable; holding, no new entries"))
            return actions
        if self._daily_loss_breached(now):
            actions.append(Action("skip", "daily loss limit hit; no entries or rotations"))
            return actions

        # 2. candidates: not held, not recently exited, not a pair that just hit a hard stop
        window = timedelta(hours=cfg.reentry_cooldown_hours)
        recent = [p for p in self.state.closed if p.closed_at and now - parse_iso(p.closed_at) < window]
        cooled = {p.pool_id for p in recent} | {p.pool_id for p in exiting.values()}
        stopped = [p for p in recent if p.close_hard] + \
                  [a.position for a in actions if a.kind == "exit" and a.hard and a.position]
        stopped_pairs = {pair_key(p.leg_symbols or p.symbol.split("-")) for p in stopped}
        held_ids = {p.pool_id for p in open_pos}
        candidates = [o for o in opps if o.pool.pool_id not in held_ids and o.pool.pool_id not in cooled
                      and pair_key(o.pool.tokens) not in stopped_pairs]
        if cfg.mode == "live":
            candidates = [o for o in candidates if tradable_live(o.pool.project)]

        equity = self.state.equity()
        cash_avail = self.state.cash_usd + sum(self._proceeds(p) for p in exiting.values()) \
            - cfg.cash_reserve_pct * equity
        remaining = [p for p in open_pos if p.position_id not in exiting]
        slots = cfg.max_positions - len(remaining)
        by_project: Dict[str, float] = {}
        by_chain: Dict[str, float] = {}
        for p in remaining:
            by_project[p.project] = by_project.get(p.project, 0.0) + p.value_usd
            by_chain[p.chain.lower()] = by_chain.get(p.chain.lower(), 0.0) + p.value_usd

        def add(d: Dict[str, float], k: str, v: float) -> None:
            d[k] = d.get(k, 0.0) + v

        # 3. entries funded by free cash and free slots
        entered: set = set()
        for o in candidates:
            if slots <= 0 or cash_avail < cfg.min_position_usd:
                break
            size = min(self._cap(o.pool, equity, by_project, by_chain), cash_avail)
            if size < cfg.min_position_usd:
                continue
            o2 = evaluate(o.pool, cfg, self.vol, size_usd=size)
            if o2.net_apr < cfg.min_net_apr:
                continue
            priced, why_not = self._entry_prices(o.pool)
            if priced is None:
                actions.append(Action("skip", f"{o.pool.symbol} on {o.pool.chain}: {why_not}; not bought"))
                continue
            actions.append(Action("enter", f"score {o2.score * 100:.1f} / net {o2.net_apr * 100:.1f}%",
                                  opportunity=o2, size_usd=size, prices=priced[0], legs=priced[1], peg=priced[2]))
            entered.add(o.pool.pool_id)
            slots -= 1
            cash_avail -= size
            add(by_project, o.pool.project, size)
            add(by_chain, o.pool.chain.lower(), size)

        # 4. rotations: best unfunded candidate against the weakest eligible holding
        def gross(p: Position) -> float:
            h = holds[p.position_id]
            return h.net_apr + h.cost_drag_apr  # forward yield before any costs

        eligible = sorted((p for p in remaining if p.position_id in holds and not p.price_stale_since
                           and hours_between(parse_iso(p.opened_at), now) >= cfg.min_hold_hours), key=gross)
        used: set = set()
        rotations = 0
        for o in candidates:
            if rotations >= cfg.max_rotations_per_cycle:
                break
            if o.pool.pool_id in entered:
                continue
            for pos in eligible:
                if pos.position_id in used:
                    continue
                proceeds = self._proceeds(pos)
                bp, bc = dict(by_project), dict(by_chain)
                add(bp, pos.project, -pos.value_usd)
                add(bc, pos.chain.lower(), -pos.value_usd)
                size = min(self._cap(o.pool, equity, bp, bc), cash_avail + proceeds)
                if size < cfg.min_position_usd:
                    continue
                o2 = evaluate(o.pool, cfg, self.vol, size_usd=size)
                held = gross(pos)
                if o2.net_apr < cfg.min_net_apr or o2.net_apr - held < cfg.rotation_margin_apr:
                    continue
                priced, _why = self._entry_prices(o.pool)
                if priced is None:
                    break
                actions.append(Action("exit", f"rotate into {o.pool.symbol} on {o.pool.chain}: "
                                      f"{o2.net_apr * 100:.1f}% net vs {held * 100:.1f}% held",
                                      position=pos, hard=False))
                actions.append(Action("enter", f"rotation from {pos.symbol} / net {o2.net_apr * 100:.1f}%",
                                      opportunity=o2, size_usd=size, prices=priced[0], legs=priced[1], peg=priced[2]))
                used.add(pos.position_id)
                entered.add(o.pool.pool_id)
                by_project, by_chain = bp, bc
                add(by_project, o.pool.project, size)
                add(by_chain, o.pool.chain.lower(), size)
                cash_avail += proceeds - size
                rotations += 1
                break
        return actions

    # --------------------------------------------------------------- execution
    def execute(self, actions: List[Action], now: datetime) -> List[Dict]:
        log: List[Dict] = []
        ordered = [a for a in actions if a.kind == "exit"] + [a for a in actions if a.kind == "enter"] + \
                  [a for a in actions if a.kind not in ("exit", "enter")]
        for a in ordered:
            if a.kind == "exit" and a.position is not None:
                pos = a.position
                fill = self.executor.close(pos)
                if not fill.ok:
                    log.append(self._entry(now, "exit-failed", pos.pool_id, pos.symbol, 0.0, fill.note))
                    continue
                pos.status = "closed"
                pos.closed_at = iso(now)
                pos.close_reason = a.reason
                pos.close_hard = a.hard
                pos.exit_cost_usd = fill.cost_usd
                pos.realised_pnl_usd = fill.size_usd - (pos.size_usd + pos.entry_cost_usd)
                self.state.cash_usd = self.executor.cash_usd()
                self.state.positions = [p for p in self.state.positions if p.position_id != pos.position_id]
                self.state.closed.append(pos)
                log.append(self._entry(now, "exit", pos.pool_id, pos.symbol, fill.size_usd,
                                       f"{a.reason}; pnl {pos.realised_pnl_usd:+.2f}", fill.tx))
            elif a.kind == "enter" and a.opportunity is not None and a.prices and a.legs:
                o = a.opportunity
                fill = self.executor.open(o, a.size_usd)
                if not fill.ok:
                    log.append(self._entry(now, "enter-failed", o.pool.pool_id, o.pool.symbol, 0.0, fill.note))
                    continue
                pa, pb = a.prices
                ka, kb, sa, sb = a.legs
                pos = Position(
                    position_id=uuid.uuid4().hex[:12], pool_id=o.pool.pool_id, chain=o.pool.chain,
                    project=o.pool.project, symbol=o.pool.symbol, opened_at=iso(now),
                    size_usd=fill.size_usd, entry_apy=o.pool.apy, entry_fee_apr=o.fee_apr,
                    entry_tvl=o.pool.tvl_usd, entry_price_ratio=pa / pb,
                    range_width=o.range_width, amp=o.amp, fee_amp=o.fee_amp, sigma=o.sigma,
                    value_usd=fill.size_usd, last_price_ratio=pa / pb,
                    last_apy=o.pool.apy, last_fee_apr=marking_fee_apr(o.pool),
                    last_reward_apr=(o.pool.apy_reward or 0.0) / 100.0 * self.cfg.reward_haircut,
                    last_tvl=o.pool.tvl_usd, last_net_apr=o.net_apr, last_hold_apr=o.net_apr,
                    last_update=iso(now), entry_cost_usd=fill.cost_usd,
                    leg_keys=[ka, kb], leg_symbols=[sa, sb], entry_prices=[pa, pb], last_prices=[pa, pb],
                    peg_legs=list(a.peg or []), price_basis_at=iso(now), drift_apr=o.drift,
                )
                self.state.positions.append(pos)
                self.state.cash_usd = self.executor.cash_usd()
                log.append(self._entry(now, "enter", o.pool.pool_id, o.pool.symbol, fill.size_usd,
                                       f"{a.reason}; cost {fill.cost_usd:.2f}", fill.tx))
            else:
                log.append(self._entry(now, a.kind, "", "", 0.0, a.reason))
        self.state.ledger.extend(log)
        return log

    @staticmethod
    def _entry(now: datetime, kind: str, pool_id: str, symbol: str, usd: float, note: str,
               tx: Optional[str] = None) -> Dict:
        return {"t": iso(now), "kind": kind, "pool_id": pool_id, "symbol": symbol,
                "usd": round(usd, 2), "note": note, "tx": tx}

    # -------------------------------------------------------------------- cycle
    def _finish(self, now: datetime, log: List[Dict], record: bool) -> Dict:
        notes, self.notes = self.notes, []
        for n in notes:
            self.state.ledger.append(self._entry(now, "note", "", "", 0.0, n))
        self.state.updated_at = iso(now)
        if record:
            self.state.equity_history.append({"t": iso(now), "equity": round(self.state.equity(), 2)})
        warnings, self.warnings = self.warnings, []
        return {"now": iso(now), "actions": log, "notes": notes, "warnings": warnings,
                "equity": self.state.equity(), "cash": self.state.cash_usd,
                "open": len(self.state.open_positions)}

    def protect(self, pools: List[Pool], now: Optional[datetime] = None, feed_up: bool = True,
                record: bool = False) -> Dict:
        """Mark every position and run only the hard exits. Runs first each cycle, before
        any slow ranking work, and alone when the yields feed is down."""
        now = now or utcnow()
        self.ctx.feed_up = feed_up
        pools_by_id = {p.pool_id: p for p in pools}
        self._mark_all(pools_by_id, now)
        log = self.execute(self.plan([], pools_by_id, now, hard_only=True), now)
        return self._finish(now, log, record)

    def tick(self, pools: List[Pool], opps: List[Opportunity], now: Optional[datetime] = None,
             feed_up: bool = True, do_mark: bool = True) -> Dict:
        """Full cycle: (mark,) every exit rule, entries, rotation, and an equity point."""
        now = now or utcnow()
        self.ctx.feed_up = feed_up
        pools_by_id = {p.pool_id: p for p in pools}
        if do_mark:
            self._mark_all(pools_by_id, now)
        held = [pools_by_id[p.pool_id] for p in self.state.open_positions if p.pool_id in pools_by_id]
        if held:
            self.vol.prefetch(held, budget_s=30.0)  # keep volatility of held pools fresh too
        log = self.execute(self.plan(opps, pools_by_id, now), now)
        return self._finish(now, log, record=True)
