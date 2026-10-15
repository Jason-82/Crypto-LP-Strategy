"""Command-line entry point.

    python3 -m lp_bot scan  [--source live|FILE] [--top 20] [--explain] [--config cfg.json]
    python3 -m lp_bot run   [--source live|FILE] [--prices live|none] [--once | --interval-min 60]
    python3 -m lp_bot status
    python3 -m lp_bot il --k 1.5 [--range 2.0]
    python3 -m lp_bot prices [coin keys...]
    python3 -m lp_bot self-test
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import traceback
from typing import Callable, List, Optional, Tuple

from .config import Config, load_config
from .executors import LiveExecutor, PaperExecutor
from .il import amplification, expected_il_rate, impermanent_loss, price_move_for_il
from .manager import Manager, utcnow
from .models import Opportunity, Pool
from .prices import DefiLlamaPrices, NullPrices, VolEstimator
from .scoring import rank
from .sources import DefiLlamaYields, FilePoolSource
from .state import InstanceLock, load_state, save_state

NETWORK_ERRORS = (OSError, ValueError, http.client.HTTPException)

DEFAULT_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "pools_sample.json")


def _source(arg: str, cfg: Config):
    if arg == "live":
        return DefiLlamaYields(cfg.yields_url, cfg.http_timeout_s)
    if arg == "fixture":
        return FilePoolSource(DEFAULT_FIXTURE)
    return FilePoolSource(arg)


def _prices(arg: str, cfg: Config):
    if arg == "live":
        return DefiLlamaPrices(cfg.prices_url, cfg.http_timeout_s)
    return NullPrices()


def _fetch(source) -> List[Pool]:
    try:
        return source.fetch()
    except NETWORK_ERRORS as exc:
        print(f"error: could not fetch pools from {source.name}: {exc}", file=sys.stderr)
        return []


def _vol(cfg: Config, prices_arg: str) -> VolEstimator:
    return VolEstimator(_prices(prices_arg, cfg), cfg.price_history_days,
                        refresh_s=cfg.vol_refresh_hours * 3600.0, retry_s=cfg.price_retry_hours * 3600.0)


def print_scan(opps: List[Opportunity], tally, top: int, explain: bool, why, pools: List[Pool],
               vol: Optional[VolEstimator] = None, cfg: Optional[Config] = None) -> None:
    print(f"{'#':>3} {'chain':<10} {'project':<24} {'pair':<22} {'tvl':>10} {'apy':>7} "
          f"{'fee':>6} {'rew':>6} {'sigma':>5} {'E[IL]':>6} {'net':>7} {'score':>7}")
    for i, o in enumerate(opps[:top], 1):
        print(f"{i:>3} {o.summary()}")
        if explain:
            extra = f"; {'; '.join(o.notes)}" if o.notes else ""
            print(f"      vol {o.vol_source} | amp {o.amp:.2f} | breakeven fee {o.breakeven_fee_apr*100:.1f}%"
                  f" | {o.pool.url}{extra}")
    print()
    print(f"{len(opps)} opportunities from {len(pools)} pools. Rejections:")
    for reason, n in tally.most_common():
        print(f"  {n:>6}  {reason}")
    if vol is not None:
        st = vol.stats()
        print(f"\nvolatility source: {st['history']} pools from price history, {st['heuristic']} from heuristics")
        for w in vol.warnings[:5]:
            print(f"  warning: {w}")
        if len(vol.warnings) > 5:
            print(f"  ... {len(vol.warnings) - 5} more warnings")
    if explain:
        floor = cfg.min_tvl_usd if cfg else 0.0
        print(f"\nHighest-APY pools above the TVL floor that were still rejected:")
        rejected = sorted((p for p in pools if p.pool_id in why and p.tvl_usd >= floor and not p.outlier),
                          key=lambda p: p.apy, reverse=True)
        for p in rejected[:15]:
            print(f"  {p.chain:<10} {p.project:<24} {p.symbol:<22} tvl ${p.tvl_usd/1e6:6.1f}M apy {p.apy:7.1f}%"
                  f"  -> {', '.join(why[p.pool_id])}")


def cmd_scan(args) -> int:
    cfg = load_config(args.config)
    if args.chains:
        cfg.chains = args.chains
    pools = _fetch(_source(args.source, cfg))
    if not pools:
        return 1
    vol = _vol(cfg, args.prices)
    opps, tally, why = rank(pools, cfg, vol, equity=cfg.starting_cash_usd)
    print_scan(opps, tally, args.top, args.explain, why, pools, vol, cfg)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([{**o.pool.to_dict(), "fee_apr": o.fee_apr, "reward_apr": o.reward_apr,
                        "sigma": o.sigma, "vol_source": o.vol_source, "amp": o.amp,
                        "range_width": o.range_width, "expected_il_apr": o.expected_il_apr,
                        "cost_drag_apr": o.cost_drag_apr, "size_usd": o.size_usd,
                        "net_apr": o.net_apr, "score": o.score, "notes": o.notes}
                       for o in opps[:args.top]], fh, indent=1)
        print(f"wrote {args.json}")
    return 0


def _build(cfg: Config, prices_arg: str):
    state = load_state(cfg.state_path, cfg.starting_cash_usd,
                       warn=lambda m: print(f"warning: {m}", file=sys.stderr))
    if cfg.mode == "live":
        executor = LiveExecutor(cfg)
        state.cash_usd = executor.cash_usd()
    else:
        executor = PaperExecutor(cfg, state.cash_usd)
    return state, executor, _vol(cfg, prices_arg)


def _sleep(seconds: float, kill_path: str) -> bool:
    """Sleep in short slices; return True early if the kill switch appears."""
    end = time.monotonic() + seconds
    while True:
        if os.path.exists(kill_path):
            return True
        left = end - time.monotonic()
        if left <= 0:
            return False
        time.sleep(min(10.0, left))


def _fetch_pools(source, cfg: Config, attempts: int, say: Callable[[str], None]) -> Tuple[List[Pool], bool]:
    """Fetch the universe with retries. Returns (pools, feed_up). A feed smaller than
    ``min_feed_pools`` is returned for marking but treated as down (no entries, no
    missing-pool clock)."""
    waits = [20, 60, 120]
    pools: List[Pool] = []
    for attempt in range(max(attempts, 1)):
        try:
            pools = source.fetch()
            if getattr(source, "skipped", 0):
                say(f"skipped {source.skipped} malformed pool records")
            if len(pools) >= cfg.min_feed_pools:
                return pools, True
            say(f"yields feed returned only {len(pools)} pools (expected {cfg.min_feed_pools}+)")
        except NETWORK_ERRORS as exc:
            say(f"yields feed attempt {attempt + 1} failed: {exc}")
        except Exception as exc:  # unexpected payload or parser bug: still protect positions
            say(f"yields feed attempt {attempt + 1} failed unexpectedly: {type(exc).__name__}: {exc}")
        if attempt < attempts - 1 and _sleep(waits[min(attempt, len(waits) - 1)], cfg.kill_switch_path):
            break
    return pools, False


def _print_report(report, extra: str = "", quiet_if_empty: bool = False) -> None:
    if quiet_if_empty and not (report["actions"] or report["notes"] or report["warnings"]):
        return
    print(f"[{report['now']}] equity ${report['equity']:,.2f} cash ${report['cash']:,.2f} "
          f"open {report['open']}{extra}")
    for a in report["actions"]:
        print(f"    {a['kind']:<12} {a['symbol']:<22} ${a['usd']:>10,.2f}  {a['note']}")
    for n in report["notes"]:
        print(f"    note         {n}")
    for w in report["warnings"]:
        print(f"    warning      {w}")


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.chains:
        cfg.chains = args.chains
    if args.source != "live":
        cfg.min_feed_pools = 1  # a file snapshot is small, but an empty one is still a dead feed
    try:
        lock = InstanceLock(cfg.state_path).__enter__()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return _run_loop(args, cfg)
    finally:
        lock.__exit__(None, None, None)


def _run_loop(args, cfg: Config) -> int:
    state, executor, vol = _build(cfg, args.prices)
    manager = Manager(cfg, state, executor, vol)
    source = _source(args.source, cfg)
    say = lambda m: print(f"    warning      {m}", file=sys.stderr)  # noqa: E731
    print(f"mode={cfg.mode} equity=${state.equity():,.2f} cash=${state.cash_usd:,.2f} "
          f"open={len(state.open_positions)} state={cfg.state_path}")
    while True:
        kill = os.path.exists(cfg.kill_switch_path)
        cycle_failed = False
        try:
            pools, feed_up = _fetch_pools(source, cfg, 1 if kill else cfg.fetch_retries, say)
            # 1. protect: mark every position and run the hard exits before anything slow
            report = manager.protect(pools, feed_up=feed_up, record=not feed_up or kill)
            save_state(cfg.state_path, state)
            _print_report(report, "" if feed_up else "  (yields feed down: holding)", quiet_if_empty=feed_up and not kill)
            # 2. full cycle: soft exits, entries, rotation
            if feed_up and not os.path.exists(cfg.kill_switch_path):
                opps, _tally, _why = rank(pools, cfg, vol, equity=state.equity(), budget_s=cfg.prefetch_budget_s)
                report = manager.tick(pools, opps, feed_up=True, do_mark=False)
                save_state(cfg.state_path, state)
                st = vol.stats()
                _print_report(report, f" candidates {len(opps)} | vol from history {st['history']}, "
                                      f"heuristic {st['heuristic']}")
                for w in vol.warnings[:5]:
                    print(f"    warning      {w}")
                del vol.warnings[:]
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep the loop alive; positions must not go unmanaged
            cycle_failed = True
            print(f"error in cycle: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=4)
            try:
                save_state(cfg.state_path, state)
            except Exception as exc2:
                print(f"error saving state: {exc2}", file=sys.stderr)
        if os.path.exists(cfg.kill_switch_path):
            if not state.open_positions:
                print("kill switch: all positions closed; stopping")
                return 0
            print(f"kill switch: {len(state.open_positions)} position(s) still open; retrying in 60s",
                  file=sys.stderr)
            if args.once:
                return 3
            time.sleep(60)
            continue
        if args.once:
            return 1 if cycle_failed else 0
        _sleep(max(args.interval_min, 1) * 60, cfg.kill_switch_path)


def cmd_status(args) -> int:
    cfg = load_config(args.config)
    state = load_state(cfg.state_path, cfg.starting_cash_usd,
                       warn=lambda m: print(f"warning: {m}", file=sys.stderr))
    eq = state.equity()
    print(f"mode={cfg.mode} updated={state.updated_at} equity=${eq:,.2f} cash=${state.cash_usd:,.2f} "
          f"vs ${cfg.starting_cash_usd:,.2f} start ({eq - cfg.starting_cash_usd:+,.2f})")
    if state.open_positions:
        print(f"\n{'pair':<16} {'chain':<9} {'project':<20} {'size':>9} {'market':>8} {'IL':>8} "
              f"{'fees':>8} {'rewards':>8} {'value':>9} {'P&L':>8} {'fwd net':>8}  flags")
        for p in state.open_positions:
            flags = []
            if p.out_of_range_since:
                flags.append("out-of-range")
            if p.price_stale_since:
                flags.append("unpriced")
            if p.missing_since:
                flags.append("missing")
            if not p.price_basis_at:
                flags.append("market-moves-not-tracked")
            print(f"{p.symbol[:16]:<16} {p.chain[:9]:<9} {p.project[:20]:<20} {p.size_usd:>9,.2f} "
                  f"{p.price_pnl_usd:>8,.2f} {p.il_usd:>8,.2f} {p.fees_accrued_usd:>8,.2f} "
                  f"{p.rewards_accrued_usd:>8,.2f} {p.value_usd:>9,.2f} {p.total_pnl_usd:>8,.2f} "
                  f"{p.last_hold_apr * 100:>7.1f}%  {' '.join(flags)}")
        print("\nmarket = what holding the two tokens made; IL = LP shortfall vs holding; "
              "P&L includes entry costs; fwd net = expected yield from here")
    if state.closed:
        realised = sum(p.realised_pnl_usd or 0.0 for p in state.closed)
        print(f"\nclosed: {len(state.closed)} positions, realised pnl ${realised:+,.2f}")
        for p in state.closed[-10:]:
            print(f"  {p.closed_at} {p.symbol:<22} {p.realised_pnl_usd:+9,.2f}  {p.close_reason}")
    notes = [r for r in state.ledger if r.get("kind") == "note"][-5:]
    if notes:
        print("\nrecent notes:")
        for r in notes:
            print(f"  {r['t']} {r['note']}")
    return 0


def cmd_il(args) -> int:
    rw = args.range
    print(f"k={args.k}  range={'full' if rw is None else rw}  amp={amplification(rw):.3f}")
    print(f"IL vs hold: {impermanent_loss(args.k, rw)*100:+.3f}%")
    if args.sigma is not None:
        print(f"expected IL/yr at sigma {args.sigma}: {expected_il_rate(args.sigma, amplification(rw))*100:.2f}%")
    if args.budget is not None:
        k_up = price_move_for_il(-abs(args.budget), rw)
        print(f"a {abs(args.budget)*100:.1f}% IL budget is reached at a {k_up:.3f}x ratio move "
              f"(or {1/k_up:.3f}x the other way)")
    return 0


def cmd_prices(args) -> int:
    """Fetch history and current prices for a few coin keys and report what came back."""
    from .prices import realised_vol
    cfg = load_config(args.config)
    src = DefiLlamaPrices(cfg.prices_url, cfg.http_timeout_s)
    series = src.history(args.keys, args.days)
    current = src.current(args.keys)
    for k in args.keys:
        pts = series.get(k, [])
        cur = current.get(k)
        cur_s = f"current {cur:.6g}" if cur else "no current price"
        if pts:
            rv = realised_vol(pts)
            vol_s = f"annualised vol {rv:.2f}" if rv is not None else "too few points for vol"
            print(f"{k}: {len(pts)} daily points, first {pts[0][1]:.6g}, last {pts[-1][1]:.6g}, {vol_s}, {cur_s}")
        else:
            print(f"{k}: no history, {cur_s}")
    for e in src.errors:
        print(f"error: {e}")
    return 0 if series and current and not src.errors else 1


def cmd_selftest(_args) -> int:
    from .selftest import run_all
    return run_all()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lp_bot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", help="JSON config file (see lp_bot/config.example.json)")
        p.add_argument("--source", default="live", help="'live' (DefiLlama), 'fixture', or a JSON file")
        p.add_argument("--prices", default="live", choices=["live", "none"])
        p.add_argument("--chains", nargs="*", help="restrict to these chains (default: all)")

    s = sub.add_parser("scan", help="rank LP opportunities across all chains")
    common(s)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--explain", action="store_true")
    s.add_argument("--json", help="write the ranked list to this file")
    s.set_defaults(fn=cmd_scan)

    r = sub.add_parser("run", help="manage positions (paper by default)")
    common(r)
    r.add_argument("--once", action="store_true", help="single cycle then exit")
    r.add_argument("--interval-min", type=float, default=60.0)
    r.set_defaults(fn=cmd_run)

    st = sub.add_parser("status", help="show positions and pnl from the state file")
    st.add_argument("--config")
    st.set_defaults(fn=cmd_status)

    il = sub.add_parser("il", help="impermanent-loss calculator")
    il.add_argument("--k", type=float, required=True, help="price ratio change, e.g. 1.5")
    il.add_argument("--range", type=float, default=None, help="symmetric range width, e.g. 2.0")
    il.add_argument("--sigma", type=float, default=None)
    il.add_argument("--budget", type=float, default=None, help="IL budget as decimal, e.g. 0.03")
    il.set_defaults(fn=cmd_il)

    pr = sub.add_parser("prices", help="debug the price-history feed for a few coin keys")
    pr.add_argument("keys", nargs="*", default=["coingecko:ethereum", "coingecko:tether",
                                                 "solana:So11111111111111111111111111111111111111112"])
    pr.add_argument("--days", type=int, default=30)
    pr.add_argument("--config")
    pr.set_defaults(fn=cmd_prices)

    t = sub.add_parser("self-test", help="run deterministic checks")
    t.set_defaults(fn=cmd_selftest)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a symbol like USD₮0 in a cp1252 console
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return args.fn(args)
