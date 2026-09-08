"""Command-line entry point.

    python3 -m lp_bot scan  [--source live|FILE] [--top 20] [--explain] [--config cfg.json]
    python3 -m lp_bot run   [--source live|FILE] [--prices live|none] [--once | --interval-min 60]
    python3 -m lp_bot status
    python3 -m lp_bot il --k 1.5 [--range 2.0]
    python3 -m lp_bot self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from .config import Config, load_config
from .executors import LiveExecutor, PaperExecutor
from .il import amplification, expected_il_rate, impermanent_loss, price_move_for_il
from .manager import Manager, utcnow
from .models import Opportunity, Pool
from .prices import DefiLlamaPrices, NullPrices, VolEstimator
from .scoring import rank
from .sources import DefiLlamaYields, FilePoolSource
from .state import load_state, save_state

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
    except Exception as exc:  # network / parse
        print(f"error: could not fetch pools from {source.name}: {exc}", file=sys.stderr)
        return []


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
    vol = VolEstimator(_prices(args.prices, cfg), cfg.price_history_days)
    opps, tally, why = rank(pools, cfg, vol)
    print_scan(opps, tally, args.top, args.explain, why, pools, vol, cfg)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([{**o.pool.to_dict(), "fee_apr": o.fee_apr, "reward_apr": o.reward_apr,
                        "sigma": o.sigma, "vol_source": o.vol_source, "amp": o.amp,
                        "expected_il_apr": o.expected_il_apr, "cost_drag_apr": o.cost_drag_apr,
                        "net_apr": o.net_apr, "score": o.score, "notes": o.notes}
                       for o in opps[:args.top]], fh, indent=1)
        print(f"wrote {args.json}")
    return 0


def _build(cfg: Config, prices_arg: str):
    state = load_state(cfg.state_path, cfg.starting_cash_usd)
    if cfg.mode == "live":
        executor = LiveExecutor(cfg)
        state.cash_usd = executor.cash_usd()
    else:
        executor = PaperExecutor(cfg, state.cash_usd)
    vol = VolEstimator(_prices(prices_arg, cfg), cfg.price_history_days)
    return state, executor, vol


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.chains:
        cfg.chains = args.chains
    state, executor, vol = _build(cfg, args.prices)
    manager = Manager(cfg, state, executor, vol)
    source = _source(args.source, cfg)
    print(f"mode={cfg.mode} equity=${state.equity():,.2f} cash=${state.cash_usd:,.2f} "
          f"open={len(state.open_positions)} state={cfg.state_path}")
    while True:
        pools = _fetch(source)
        if pools:
            opps, tally, _why = rank(pools, cfg, vol)
            report = manager.tick(pools, opps)
            save_state(cfg.state_path, state)
            print(f"[{report['now']}] equity ${report['equity']:,.2f} cash ${report['cash']:,.2f} "
                  f"open {report['open']} candidates {len(opps)}")
            for a in report["actions"]:
                print(f"    {a['kind']:<12} {a['symbol']:<22} ${a['usd']:>10,.2f}  {a['note']}")
        if args.once or os.path.exists(cfg.kill_switch_path):
            break
        time.sleep(max(args.interval_min, 1) * 60)
    return 0


def cmd_status(args) -> int:
    cfg = load_config(args.config)
    state = load_state(cfg.state_path, cfg.starting_cash_usd)
    print(f"mode={cfg.mode} updated={state.updated_at} equity=${state.equity():,.2f} "
          f"cash=${state.cash_usd:,.2f}")
    if state.open_positions:
        print(f"\n{'pair':<22} {'chain':<10} {'project':<22} {'size':>10} {'fees':>9} {'IL':>9} "
              f"{'value':>10} {'apy':>6} {'opened':<20}")
        for p in state.open_positions:
            print(f"{p.symbol:<22} {p.chain:<10} {p.project:<22} {p.size_usd:>10,.2f} "
                  f"{p.fees_accrued_usd:>9,.2f} {p.il_usd:>9,.2f} {p.value_usd:>10,.2f} "
                  f"{p.last_apy:>6.1f} {p.opened_at:<20}")
    if state.closed:
        realised = sum(p.realised_pnl_usd or 0.0 for p in state.closed)
        print(f"\nclosed: {len(state.closed)} positions, realised pnl ${realised:+,.2f}")
        for p in state.closed[-10:]:
            print(f"  {p.closed_at} {p.symbol:<22} {p.realised_pnl_usd:+9,.2f}  {p.close_reason}")
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

    t = sub.add_parser("self-test", help="run deterministic checks")
    t.set_defaults(fn=cmd_selftest)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
