"""Deterministic checks, no network. Each block pins a behaviour the live review found wrong
or that the strategy depends on: IL arithmetic, classification, price alignment and caching,
scoring, USD valuation, fee and gap accounting, every exit rule, rotation, persistence, and
the run-loop safety rails."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import replace as dataclasses_replace
from datetime import datetime, timedelta, timezone

from .config import Config, load_config
from .executors import PaperExecutor
from .il import (amplification, expected_il_rate, hold_value, il_constant_product, impermanent_loss,
                 in_range, pair_sigma, position_value, price_move_for_il)
from .manager import Manager, iso
from .models import Pool, Position
from .prices import (FixturePrices, NullPrices, VolEstimator, _remap_keys, chain_key, classify,
                     heuristic_sigma, pair_key, price_keys, ratio_sigma_from_history, realised_vol)
from .scoring import (conservative_fee_apr, evaluate, is_concentrated, marking_fee_apr, rank,
                      ranking_size, reject_reasons)
from .sources import FilePoolSource, parse_llama_pool, parse_records
from .state import InstanceLock, State, load_state, save_state

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "pools_sample.json")
T0 = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
H = timedelta(hours=1)

_fails = []


def check(cond, msg):
    if not cond:
        _fails.append(msg)
        print(f"  FAIL  {msg}")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ----------------------------------------------------------------------------- helpers
def gbm(seed, start, sigma, n=120):
    rng = random.Random(seed)
    s = sigma / math.sqrt(365)
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * math.exp(rng.gauss(0, s)))
    return out


def flat(seed, level=1.0, n=120, noise=0.0002):
    rng = random.Random(seed)
    return [level * (1 + rng.gauss(0, noise)) for _ in range(n)]


def series():
    return {
        "coingecko:ethereum": gbm(1, 2500.0, 0.7),
        "coingecko:solana": gbm(2, 150.0, 0.8),
        "coingecko:wrapped-bitcoin": gbm(3, 60000.0, 0.55),
        "coingecko:usd-coin": flat(4),
        "coingecko:tether": flat(5),
    }


def pool(pid, chain, project, symbol, apy, base=None, b7=None, mean=None, reward=None, tvl=50e6, age=400):
    return Pool(pid, chain, project, symbol, tvl, apy, apy_base=apy if base is None else base,
                apy_reward=reward, apy_base_7d=b7, apy_mean_30d=mean, age_days=age)


def mini_universe():
    """Two volatile pairs on cheap chains plus a stable pair."""
    return [
        pool("eth-arb", "Arbitrum", "uniswap-v3", "WETH-USDC", 40.0, b7=38.0, mean=36.0),
        pool("sol-ray", "Solana", "raydium-amm", "SOL-USDC", 45.0, b7=44.0, mean=42.0),
        pool("stab-arb", "Arbitrum", "uniswap-v3", "USDT-USDC", 14.0, b7=13.0, mean=13.5),
    ]


def make(tmp, prices=None, cash=20_000.0, **over):
    over.setdefault("min_tvl_usd", 1e6)
    cfg = Config(state_path=os.path.join(tmp, "state.json"), kill_switch_path=os.path.join(tmp, "STOP"),
                 starting_cash_usd=cash, **over)
    state = State(cash_usd=cash)
    ex = PaperExecutor(cfg, cash)
    px = prices if prices is not None else FixturePrices(series())
    vol = VolEstimator(px, 90)
    return cfg, state, ex, vol, Manager(cfg, state, ex, vol), px


def ranked(mgr, pools):
    return rank(pools, mgr.cfg, mgr.vol, equity=mgr.state.equity())[0]


def move(px, key, factor):
    px.series[key][-1] *= factor


def by_pool(state, pid):
    return next((p for p in state.open_positions if p.pool_id == pid), None)


# ----------------------------------------------------------------------------- tests
def test_il():
    check(close(il_constant_product(4.0), -0.2), "IL at 4x should be -20%")
    check(close(il_constant_product(2.0), il_constant_product(0.5)), "IL symmetric in log space")
    check(close(amplification(4.0), 2.0) and close(amplification(None), 1.0), "amplification")
    check(close(impermanent_loss(1.5, 2.0), amplification(2.0) * il_constant_product(1.5)), "in-range CL IL")
    check(close(position_value(2.0, 2.0), position_value(2.0000001, 2.0), 1e-6), "value continuous at top")
    check(close(position_value(0.5, 2.0), position_value(0.4999999, 2.0), 1e-6), "value continuous at bottom")
    check(in_range(1.9, 2.0) and not in_range(2.1, 2.0) and in_range(9.0, None), "in_range")
    check(close(expected_il_rate(0.7, 1.0), 0.06125), "E[IL] = sigma^2/8")
    check(close(price_move_for_il(-0.2, None), 4.0, 1e-6), "IL budget inversion full range")
    check(close(pair_sigma(0.6, 0.6, 1.0), 0.0), "pair sigma")


def test_classification():
    cases = {"WETH.E": "eth", "WBTC.B": "btc", "BTC.B": "btc", "FXRP": "major", "MSUSD": "stable",
             "MSETH": "eth", "JITOSOL": "sol", "USDCAD": "fx", "USDCHF": "fx", "USD₮0": "stable",
             "usdc.e": "stable", "EURC": "eurstable", "PENGU": "exotic", "SOL": "sol"}
    for sym, want in cases.items():
        check(classify(sym) == want, f"classify {sym} -> {want} (got {classify(sym)})")
    check(close(heuristic_sigma(["JITOSOL", "SOL"])[0], 0.06), "LST vs SOL is a correlated pair")
    check(close(heuristic_sigma(["EURC", "USDC"])[0], 0.08), "euro vs dollar stable carries FX vol")
    check(heuristic_sigma(["WETH.E", "USDC"])[0] == heuristic_sigma(["WETH", "USDC"])[0], "bridged ETH = ETH")
    check(pair_key(["WETH", "USDC"]) == pair_key(["ETH", "USDC.E"]), "pair key normalises wrappers")
    check(chain_key("OP Mainnet") == "optimism" and chain_key("Avalanche") == "avax"
          and chain_key("ZKsync Era") == "era" and chain_key("Ethereum") == "ethereum", "coins-API chain keys")
    p = parse_llama_pool({"pool": "a", "chain": "Avalanche", "project": "pharaoh-v3", "symbol": "WAVAX-USDC",
                          "tvlUsd": 1, "apy": 1, "underlyingTokens": ["0xAbC", "0xdef"]})
    check(price_keys(p) == ["avax:0xAbC", "avax:0xdef"], f"price keys use coins chain key ({price_keys(p)})")


def test_price_alignment_and_cache():
    rng = random.Random(11)
    a, b = [100.0], [100.0]
    for _ in range(120):
        common = rng.gauss(0, 0.03)
        a.append(a[-1] * math.exp(common + rng.gauss(0, 0.002)))
        b.append(b[-1] * math.exp(common + rng.gauss(0, 0.002)))
    da = [(1000 + i, p) for i, p in enumerate(a)]
    db = [(1000 + i, p) for i, p in enumerate(b)]
    base = ratio_sigma_from_history(da, db)
    check(base is not None and base < 0.1, f"correlated legs give low ratio vol ({base})")
    db_gap = [x for x in db if x[0] != 1050]   # one leg misses a day
    db_short = db[:-1]                          # one leg lacks the latest day (other endpoint)
    for name, other in (("missing day", db_gap), ("shorter series", db_short)):
        s = ratio_sigma_from_history(da, other)
        check(s is not None and abs(s - base) < 0.02, f"{name} does not shift a leg ({s} vs {base})")
    check(realised_vol([1.0] * 5) is None, "too-short history -> None")
    check(_remap_keys(["arbitrum:0xAbC"], {"arbitrum:0xabc": 1}) == {"arbitrum:0xAbC": 1}, "case-insensitive keys")

    clock = [0.0]

    class Flaky(FixturePrices):
        calls = 0

        def history(self, keys, days, deadline=None):
            Flaky.calls += 1
            return {} if Flaky.calls == 1 else super().history(keys, days)

    vol = VolEstimator(Flaky(series()), 90, refresh_s=86400, retry_s=6 * 3600, clock=lambda: clock[0])
    p = pool("e", "Arbitrum", "uniswap-v3", "WETH-USDC", 30.0)
    _, src = vol.estimate(p)
    check(src.startswith("heuristic") and Flaky.calls == 0, "estimate never fetches on its own")
    vol.prefetch([p])
    check(vol.estimate(p)[1].startswith("heuristic") and Flaky.calls == 1, "first fetch fails -> heuristic")
    clock[0] = 3600.0
    vol.prefetch([p])
    check(vol.estimate(p)[1].startswith("heuristic") and Flaky.calls == 1, "failed coins not retried before retry_s")
    clock[0] = 7 * 3600.0
    vol.prefetch([p])
    check(vol.estimate(p)[1] == "history", "failed coins retried after retry_s")
    good = dict(vol._hist)
    vol.prices.fail_history = True
    clock[0] = 40 * 3600.0
    vol.prefetch([p])
    check(vol.estimate(p)[1] == "history" and vol._hist["coingecko:ethereum"][1] == good["coingecko:ethereum"][1],
          "a failed refresh keeps the last good history")


def test_sources_and_scoring():
    pools, skipped = parse_records([{"pool": "x", "chain": "Base", "project": "uniswap-v3", "symbol": "A-B",
                                     "tvlUsd": "5e6", "apy": 10, "count": "12.0"},
                                    {"pool": None}, "garbage", {"pool": "y", "tvlUsd": {"bad": 1}}])
    check(len(pools) == 2 and skipped == 2 and pools[0].age_days == 12, f"per-record parsing ({len(pools)}, {skipped})")
    for slug, want in {"orca-dex": True, "cetus-amm": True, "hyperion": True, "trader-joe-v2.1": True,
                       "sparkdex-v4": True, "uniswap-v4": True, "raydium-clmm": True, "balancer-v3": False,
                       "curve-dex": False, "uniswap-v2": False, "raydium-amm": False, "aerodrome-v1": False}.items():
        check(is_concentrated(Pool("x", "c", slug, "A-B", 1, 1)) == want, f"concentrated({slug}) == {want}")
    no_base = Pool("n", "Ethereum", "uniswap-v3", "A-B", 1e7, 100.0, apy_base=None, apy_base_7d=150.0, apy_mean_30d=200.0)
    check(conservative_fee_apr(no_base)[0] <= 1.0, "fee APR never exceeds total APY")
    dead = Pool("d", "Ethereum", "uniswap-v3", "A-B", 1e7, 50.0, apy_base=50.0, apy_mean_30d=0.0)
    check(close(conservative_fee_apr(dead)[0], 0.0), "a 0% 30d mean caps a spike to zero")
    no30 = Pool("h", "Ethereum", "uniswap-v3", "A-B", 1e7, 100.0, apy_base=100.0)
    check(close(conservative_fee_apr(no30)[0], 0.50), "no 30d mean halves the fee APR")
    spiky = Pool("s", "Ethereum", "uniswap-v3", "A-B", 1e7, 90.0, apy_base=90.0, apy_base_7d=40.0, apy_mean_30d=40.0)
    check(close(conservative_fee_apr(spiky, smooth=True)[0], 0.40) and close(marking_fee_apr(spiky), 0.90),
          "holding uses the 7-day fee; marking uses today's")
    cfg = Config()
    check("pool age unknown" in reject_reasons(pool("u", "Base", "uniswap-v3", "WETH-USDC", 30.0, age=None), cfg),
          "unknown pool age is rejected by default")
    check(close(ranking_size(cfg, pool("r", "Base", "uniswap-v3", "WETH-USDC", 30.0), 25_000.0), 5_000.0),
          "ranking size follows the 20% cap, not max_position_usd")
    vol = VolEstimator(FixturePrices(series()), 90)
    st = evaluate(pool("st", "Arbitrum", "uniswap-v3", "USDT-USDC", 14.0, b7=13.0, mean=13.5), cfg, vol)
    check(st.stable_pair and st.range_width is not None and st.range_width < 1.05, "stable CL pair gets a tight range")
    st_home = evaluate(pool("h", "Arbitrum", "uniswap-v2", "WETH-USDC", 30.0, b7=30.0, mean=30.0), cfg, vol, size_usd=5000)
    st_away = evaluate(pool("a", "Base", "uniswap-v2", "WETH-USDC", 30.0, b7=30.0, mean=30.0), cfg, vol, size_usd=5000)
    check(st_away.cost_drag_apr > st_home.cost_drag_apr, "a non-home chain pays bridge costs")

    fx = FilePoolSource(FIXTURE).fetch()
    opps, tally, why = rank(fx, cfg, VolEstimator(NullPrices(), 90), equity=25_000.0)
    for pid, reason in {"fx-eth-scam-farm": "apy implausibly high", "fx-eth-aave-usdc": "single-asset (not an LP pair)",
                        "fx-base-uni3-usdc-cbbtc": "pool too young", "fx-base-uni3-brett-weth": "expected IL too large",
                        "fx-arb-camelot3-grail-weth": "reward-emission heavy"}.items():
        check(reason in why.get(pid, []), f"{pid} rejected for '{reason}' (got {why.get(pid)})")
    check(opps and all(o.net_apr >= cfg.min_net_apr for o in opps), "survivors clear the net floor")
    check([o.score for o in opps] == sorted((o.score for o in opps), reverse=True), "ranked by score")


def test_entries_and_valuation():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp, prices=NullPrices())
        uni = mini_universe()
        rep = mgr.tick(uni, ranked(mgr, uni), now=T0)
        check(not state.open_positions and any("price" in a["note"] for a in rep["actions"]),
              "no entry without prices for both tokens")

    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        uni = mini_universe()
        mgr.tick(uni, ranked(mgr, uni), now=T0)
        check(len(state.open_positions) == 3, f"entered all three ({len(state.open_positions)})")
        check(all(p.size_usd <= 0.2 * 20_000 for p in state.open_positions), "per-pool cap")
        check(state.cash_usd >= cfg.cash_reserve_pct * state.equity() - 1e-6, "cash reserve kept")
        eth = by_pool(state, "eth-arb")
        stab = by_pool(state, "stab-arb")
        check(eth.leg_symbols == ["WETH", "USDC"] and eth.entry_prices[1] > 0.99 and eth.peg_legs == [1],
              "entry records legs, USD prices and the stablecoin leg")
        check(stab.peg_legs == [0, 1] and stab.range_width < 1.05, "stable pair: both legs pegged, tight range")
        bridge = by_pool(state, "sol-ray")
        check(bridge.entry_cost_usd > eth.entry_cost_usd, "Solana entry pays a bridge hop from the home chain")

        # ETH +20% two hours later: market P&L, IL and value follow the formulas
        move(px, "coingecko:ethereum", 1.2)
        mgr.protect(uni, now=T0 + 2 * H)
        k, qb = mgr._k_qb(eth)
        check(close(k, 1.2 * px.series["coingecko:usd-coin"][-1] / px.series["coingecko:usd-coin"][-1], 1e-9), "k from USD prices")
        exp_hold = eth.size_usd * hold_value(k) * qb
        exp_lp = eth.size_usd * position_value(k, eth.range_width) * qb
        check(close(eth.price_pnl_usd, exp_hold - eth.size_usd, 1e-6), "market P&L = holding the two tokens")
        check(close(eth.il_usd, exp_lp - exp_hold, 1e-6) and eth.il_usd < 0, "IL = LP vs holding")
        fee = exp_lp * marking_fee_apr(uni[0]) * 2 / 8760
        check(close(eth.fees_accrued_usd, fee, 1e-6), "fees accrue at today's rate on the liquidity's current value")
        check(close(eth.value_usd, exp_lp + eth.fees_accrued_usd + eth.rewards_accrued_usd, 1e-6), "value = LP + fees")
        check(close(state.equity(), state.cash_usd + sum(p.value_usd for p in state.open_positions)), "equity in USD")


def test_rewards_and_gaps():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        uni = [pool("rw", "Arbitrum", "uniswap-v2", "WETH-USDC", 40.0, base=30.0, reward=10.0, b7=30.0, mean=40.0)]
        mgr.tick(uni, ranked(mgr, uni), now=T0)
        p = state.open_positions[0]
        mgr.protect(uni, now=T0 + 2 * H)
        check(close(p.rewards_accrued_usd, p.size_usd * 0.10 * cfg.reward_haircut * 2 / 8760, 1e-6),
              "rewards accrue at the haircut rate the entry was scored on")
        before = p.fees_accrued_usd
        lower = [pool("rw", "Arbitrum", "uniswap-v2", "WETH-USDC", 15.0, base=12.0, reward=3.0, b7=20.0, mean=30.0)]
        rep = mgr.protect(lower, now=T0 + 2 * H + 20 * 24 * H)
        check(close(p.fees_accrued_usd - before, p.size_usd * 0.12 * 20 * 24 / 8760, 1e-6),
              "a 20-day gap is booked at the lower of the before/after rates")
        check(p.gap_hours >= 480 and any("gap" in n for n in rep["notes"]), "gap recorded and reported")


def test_soft_exits():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        spike = [pool("sp", "Solana", "raydium-amm", "SOL-USDC", 98.6, b7=60.0, mean=45.0)]
        mgr.tick(spike, ranked(mgr, spike), now=T0)
        p = state.open_positions[0]
        check(close(p.entry_fee_apr, 0.60), "bought on the conservative fee, not the 1-day spike")
        normal = [pool("sp", "Solana", "raydium-amm", "SOL-USDC", 30.0, b7=45.0, mean=45.0)]
        rep = mgr.tick(normal, [], now=T0 + 72 * H)
        check(not any(a["kind"] == "exit" for a in rep["actions"]),
              f"spike normalising is not an exit ({[a['note'] for a in rep['actions']]})")
        quiet = [pool("sp", "Solana", "raydium-amm", "SOL-USDC", 5.0, b7=44.0, mean=45.0)]
        rep = mgr.tick(quiet, [], now=T0 + 73 * H)
        check(not any(a["kind"] == "exit" for a in rep["actions"]), "one quiet day is not an exit")
        dead = [pool("sp", "Solana", "raydium-amm", "SOL-USDC", 6.0, b7=7.0, mean=20.0)]
        early = mgr.exit_reason(p, dead[0], T0 + 24 * H)
        check(early is None, "soft exits wait for the minimum hold")
        rep = mgr.tick(dead, [], now=T0 + 74 * H)
        ex_ = [a for a in rep["actions"] if a["kind"] == "exit"]
        check(len(ex_) == 1 and "expected to lose money" in ex_[0]["note"], f"collapsed fees exit on economics ({ex_})")

    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        uni = [pool("tv", "Arbitrum", "uniswap-v2", "WETH-USDC", 40.0, b7=40.0, mean=40.0, tvl=40e6)]
        mgr.tick(uni, ranked(mgr, uni), now=T0)
        p = state.open_positions[0]
        move(px, "coingecko:ethereum", 0.5)   # ETH halves; TVL falls with it, liquidity did not leave
        crash = [pool("tv", "Arbitrum", "uniswap-v2", "WETH-USDC", 40.0, b7=40.0, mean=40.0, tvl=30e6)]
        mgr.protect(crash, now=T0 + 72 * H)
        r = mgr.exit_reason(p, crash[0], T0 + 72 * H, hard_only=False)
        check(r is None or "liquidity" not in r[0], f"a price crash alone is not a TVL-drain exit ({r})")
        check(p.price_pnl_usd < -0.2 * p.size_usd, "the crash shows up as a market loss in USD")


def test_hard_exits():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        uni = mini_universe()
        mgr.tick(uni, ranked(mgr, uni), now=T0)
        # IL budget on the concentrated ETH position; the sibling ETH/USDC pool is not bought this cycle
        move(px, "coingecko:ethereum", 1.6)
        sibling = uni + [pool("eth-base", "Base", "uniswap-v3", "WETH-USDC", 60.0, b7=58.0, mean=55.0)]
        rep = mgr.tick(sibling, ranked(mgr, sibling), now=T0 + 1 * H)
        exits = [a for a in rep["actions"] if a["kind"] == "exit"]
        check(any(a["pool_id"] == "eth-arb" and "IL budget" in a["note"] for a in exits), f"IL stop ({exits})")
        check(by_pool(state, "eth-base") is None, "no re-buying the same pair right after its stop")
        # de-peg on the stable pair
        move(px, "coingecko:tether", 0.975)
        rep = mgr.protect(uni, now=T0 + 2 * H)
        check(any(a["pool_id"] == "stab-arb" and "peg" in a["note"] for a in rep["actions"]), "de-peg stop")

    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        uni = mini_universe()
        mgr.tick(uni, ranked(mgr, uni), now=T0)
        sol = by_pool(state, "sol-ray")
        # pool vanishes from a healthy feed: no fees, grace period, then exit
        others = [p for p in uni if p.pool_id != "sol-ray"]
        mgr.protect(others, now=T0 + 1 * H)
        fees = sol.fees_accrued_usd
        r = mgr.protect(others, now=T0 + 3 * H)
        check(sol.status == "open" and close(sol.fees_accrued_usd, fees), "missing pool: no fees, no instant exit")
        r = mgr.protect(others, now=T0 + 26 * H)
        check(any(a["pool_id"] == "sol-ray" and "missing" in a["note"] for a in r["actions"]), "missing pool exits after grace")
        # whole feed down: nothing exits on missing data, nothing is bought
        eth = by_pool(state, "eth-arb")
        rep = mgr.protect([], now=T0 + 27 * H, feed_up=False)
        check(eth.status == "open" and eth.missing_since is None, "feed outage does not start the missing clock")
        rep = mgr.tick([], ranked(mgr, uni), now=T0 + 28 * H, feed_up=False)
        check(not any(a["kind"] == "enter" for a in rep["actions"]), "no entries while the feed is down")

    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        uni = mini_universe()
        mgr.tick(uni, ranked(mgr, uni), now=T0)
        sol_series = px.series.pop("coingecko:solana")
        mgr.protect(uni, now=T0 + 1 * H)
        sol = by_pool(state, "sol-ray")
        check(sol.price_stale_since is not None and sol.status == "open", "unpriced position is flagged, not sold")
        rep = mgr.protect(uni, now=T0 + 8 * H)
        check(any(a["pool_id"] == "sol-ray" and "could not price" in a["note"] for a in rep["actions"]),
              "position unpriceable for longer than the grace period is closed")
        px.series["coingecko:solana"] = sol_series
        px.fail_current = True
        rep = mgr.protect(uni, now=T0 + 20 * H)
        check(not any(a["kind"] == "exit" for a in rep["actions"]), "a price-feed outage does not liquidate")

    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp, il_stop_pct=0.99, depeg_stop_pct=0.9)
        uni = mini_universe()
        mgr.tick(uni, ranked(mgr, uni), now=T0)
        move(px, "coingecko:ethereum", 3.5)
        rep = mgr.protect(uni, now=T0 + 1 * H)
        eth = by_pool(state, "eth-arb")
        check(eth.out_of_range_since is not None and eth.status == "open", "out of range starts the grace clock")
        rep = mgr.protect(uni, now=T0 + 14 * H)
        check(any(a["pool_id"] == "eth-arb" and "out of range" in a["note"] for a in rep["actions"]), "out-of-range exit")
        # kill switch with the feed down closes everything
        open(cfg.kill_switch_path, "w").close()
        rep = mgr.protect([], now=T0 + 15 * H, feed_up=False)
        check(not state.open_positions, "kill switch closes everything even when the feed is down")
        os.unlink(cfg.kill_switch_path)


def test_rotation_and_loss_limit():
    strong = pool("strong", "Solana", "raydium-amm", "SOL-USDC", 95.0, b7=90.0, mean=88.0)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp, max_positions=3)
        held = mini_universe()
        mgr.tick(held, ranked(mgr, held), now=T0)
        check(len(state.open_positions) == 3, "three positions held, no free slot")
        both = held + [strong]
        rep = mgr.tick(both, ranked(mgr, both), now=T0 + 1 * H)
        check(not any(a["kind"] == "exit" for a in rep["actions"]), "no rotation inside the minimum hold")
        rep = mgr.tick(both, ranked(mgr, both), now=T0 + 72 * H)
        exits = [a for a in rep["actions"] if a["kind"] == "exit"]
        enters = [a for a in rep["actions"] if a["kind"] == "enter"]
        check(len(exits) == 1 and len(enters) == 1 and enters[0]["pool_id"] == "strong",
              f"exactly one paired rotation into the better pool ({[a['note'] for a in rep['actions']]})")
        check(len(state.open_positions) == 3, "rotation keeps the slot count")
        rotated = exits[0]["pool_id"] if exits else None
        weakest = min((p for p in held), key=lambda p: evaluate(p, cfg, vol, holding=True).net_apr).pool_id
        check(rotated == weakest, f"the weakest holding is the one rotated ({rotated} vs {weakest})")

    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp, max_positions=3)
        held = mini_universe()
        mgr.tick(held, ranked(mgr, held), now=T0)
        state.equity_history.append({"t": iso(T0 + 47 * H), "equity": state.equity() * 1.2})
        both = held + [strong]
        rep = mgr.tick(both, ranked(mgr, both), now=T0 + 72 * H)
        check(not any(a["kind"] in ("exit", "enter") for a in rep["actions"])
              and any("daily loss" in a["note"] for a in rep["actions"]), "loss limit blocks rotations too")

    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        state.equity_history.append({"t": iso(T0 - 20 * 24 * H), "equity": 40_000.0})
        check(not mgr._daily_loss_breached(T0), "a 20-day-old equity point is not a daily reference")


def test_legacy_migration():
    legacy = {"position_id": "old1", "pool_id": "eth-arb", "chain": "Arbitrum", "project": "uniswap-v3",
              "symbol": "WETH-USDC", "opened_at": iso(T0 - 24 * H), "size_usd": 5000.0, "entry_apy": 40.0,
              "entry_fee_apr": 0.38, "entry_tvl": 50e6, "entry_price_ratio": 2400.0, "range_width": 3.0,
              "amp": 2.366, "fee_amp": 1.0, "sigma": 0.7, "fees_accrued_usd": 12.0, "il_usd": -1.0,
              "value_usd": 5011.0, "last_price_ratio": 2450.0, "last_fee_apr": 0.38,
              "last_update": iso(T0 - 1 * H), "status": "open"}
    pos = Position.from_dict(dict(legacy, unknown_future_field=1))
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        state.positions.append(pos)
        uni = mini_universe()
        mgr.protect(uni, now=T0)
        eth_usd = px.series["coingecko:ethereum"][-1] / px.series["coingecko:usd-coin"][-1]
        check(pos.entry_prices and pos.price_basis_at and pos.leg_keys, "legacy position migrated to USD valuation")
        check(close(mgr._k_qb(pos)[0], eth_usd / 2400.0, 1e-9), "migration keeps the original entry ratio for IL")
        placeholder = Position.from_dict(dict(legacy, position_id="old2", entry_price_ratio=1.0, last_price_ratio=1.0))
        state.positions.append(placeholder)
        rep = mgr.protect(uni, now=T0 + 1 * H)
        check(placeholder.status == "open" and abs(placeholder.il_usd) < 1e-6
              and any("never recorded" in n for n in rep["notes"]), "a 1.0 placeholder ratio is rebased, not a phantom loss")


def test_persistence_and_lock():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.json")
        st = State(cash_usd=123.0)
        save_state(path, st)
        save_state(path, State(cash_usd=456.0))
        check(load_state(path, 0.0).cash_usd == 456.0 and os.path.exists(path + ".bak"), "save keeps a .bak")
        with open(path, "w") as fh:
            fh.write("")
        check(load_state(path, 0.0).cash_usd == 123.0, "corrupt state falls back to .bak")
        real_replace = os.replace
        calls = {"n": 0}

        def flaky_replace(a, b):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError("file in use")
            return real_replace(a, b)

        os.replace = flaky_replace
        try:
            save_state(path, State(cash_usd=789.0))
        finally:
            os.replace = real_replace
        check(load_state(path, 0.0).cash_usd == 789.0 and calls["n"] == 3, "save retries a locked rename")
        with InstanceLock(path):
            try:
                with InstanceLock(path):
                    check(False, "a second run must not get the lock")
            except RuntimeError:
                pass
        with InstanceLock(path):
            pass  # released and re-acquirable


def test_config():
    cfg = load_config(os.path.join(HERE, "config.example.json"))
    check(cfg.mode == "paper" and cfg.gas_for("Ethereum") == 60 and cfg.gas_for("zksync") == 3.0, "example config loads")
    check("gmx-v2-perps" in cfg.project_blocklist and "CRCLX" in cfg.token_blocklist, "default blocklists kept")
    old = {"_blocklists": "x", "mode": "paper", "apy_decay_ratio": 0.5, "max_share_of_pool_tvl": 0.005,
           "stable_sigma_threshold": 0.15, "cl_fee_amp_share": 0.0, "reentry_cooldown_hours": 24}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(old, fh)
    try:
        check(load_config(fh.name).apy_decay_ratio == 0.5, "a config written for the previous version still loads")
    finally:
        os.unlink(fh.name)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"bogus": 1}, fh)
    try:
        load_config(fh.name)
        check(False, "unknown key should raise")
    except ValueError:
        pass
    finally:
        os.unlink(fh.name)


def test_review_round_two():
    # --- migration of a stable-first legacy pair keeps the real basis and does not de-peg
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        eth_now = px.series["coingecko:ethereum"][-1]
        legacy = Position.from_dict({
            "position_id": "sf", "pool_id": "usdc-weth", "chain": "Arbitrum", "project": "uniswap-v3",
            "symbol": "USDC-WETH", "opened_at": iso(T0 - 24 * H), "size_usd": 5000.0, "entry_apy": 30.0,
            "entry_fee_apr": 0.3, "entry_tvl": 50e6, "entry_price_ratio": 1 / (eth_now / 1.04), "range_width": 3.0,
            "amp": 2.366, "fee_amp": 1.0, "sigma": 0.7, "value_usd": 5000.0, "last_price_ratio": 1 / eth_now,
            "last_fee_apr": 0.3, "last_update": iso(T0 - 1 * H), "status": "open"})
        state.positions.append(legacy)
        state.equity_history.append({"t": iso(T0 - 2 * H), "equity": 20_000.0})
        uni = [pool("usdc-weth", "Arbitrum", "uniswap-v3", "USDC-WETH", 30.0, b7=30.0, mean=30.0)]
        rep = mgr.protect(uni, now=T0)
        check(legacy.status == "open" and close(legacy.entry_prices[0], 1.0) and legacy.peg_legs == [0],
              f"stable-first legacy pair anchored on the stablecoin ({legacy.entry_prices}, {legacy.peg_legs})")
        check(legacy.price_pnl_usd > 0, "ETH up ~4% since entry shows as a market gain, not a loss")
        check(state.equity_history[-1]["equity"] > 20_000.0 + 0.5 * legacy.price_pnl_usd,
              "the one-off revaluation is netted out of the daily-loss history")

    # --- placeholder within 1.5x is replaced by the historical ratio; non-stable quote gets a real basis
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        btc0, eth0 = 50000.0, 2000.0
        px.at[int((T0 - 48 * H).timestamp())] = {"coingecko:wrapped-bitcoin": btc0, "coingecko:ethereum": eth0}
        ph = Position.from_dict({
            "position_id": "ph", "pool_id": "wbtc-weth", "chain": "Arbitrum", "project": "uniswap-v2",
            "symbol": "WBTC-WETH", "opened_at": iso(T0 - 48 * H), "size_usd": 5000.0, "entry_apy": 20.0,
            "entry_fee_apr": 0.2, "entry_tvl": 50e6, "entry_price_ratio": 1.0, "range_width": None, "amp": 1.0,
            "fee_amp": 1.0, "sigma": 0.4, "value_usd": 5000.0, "last_price_ratio": 1.0, "last_fee_apr": 0.2,
            "last_update": iso(T0 - 1 * H), "status": "open"})
        state.positions.append(ph)
        mgr.protect([pool("wbtc-weth", "Arbitrum", "uniswap-v2", "WBTC-WETH", 20.0, b7=20.0, mean=20.0)], now=T0)
        check(close(ph.entry_price_ratio, btc0 / eth0) and close(ph.entry_prices[1], eth0),
              f"placeholder ratio recovered from history at entry ({ph.entry_price_ratio}, {ph.entry_prices})")

    # --- one unpriceable holding: entries continue, and it is closed after the grace period
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        odd = Position.from_dict({
            "position_id": "odd", "pool_id": "pengu", "chain": "Solana", "project": "orca-dex",
            "symbol": "PENGU-WETH", "opened_at": iso(T0 - 24 * H), "size_usd": 3000.0, "entry_apy": 50.0,
            "entry_fee_apr": 0.3, "entry_tvl": 9e6, "entry_price_ratio": 1.0, "range_width": None, "amp": 1.0, "fee_amp": 1.0,
            "sigma": 1.2, "value_usd": 3000.0, "last_price_ratio": 1.0, "last_update": iso(T0 - 1 * H), "status": "open"})
        state.positions.append(odd)
        state.cash_usd -= 3000.0
        ex._cash -= 3000.0
        uni = mini_universe() + [pool("pengu", "Solana", "orca-dex", "PENGU-WETH", 50.0, b7=50.0, mean=50.0)]
        rep = mgr.tick(uni, ranked(mgr, uni), now=T0)
        check(any(a["kind"] == "enter" for a in rep["actions"]), "one unpriceable holding does not block entries")
        rep = mgr.protect(uni, now=T0 + 7 * H)
        check(odd.status == "closed" and "could not price" in (odd.close_reason or ""), "unpriceable holding closed after grace")

    # --- a rejected coin in a batch does not blank the others; casings; breaker ignores bad keys
    from .prices import DefiLlamaPrices, HttpStatusError

    class FakeLlama(DefiLlamaPrices):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def _get(self, path):
            self.calls += 1
            if "bad:" in path:
                raise HttpStatusError(400, "bad coin")
            coins = path.split("/")[-1].split("?")[0].split(",")
            return {"coins": {c.replace("%3A", ":").upper() if c.startswith("x") else c: {"price": 1.5} for c in coins}}

        def _chart(self, keys, days):
            if any(k.startswith("bad:") for k in keys):
                raise HttpStatusError(400, "bad coin")
            return {k: [(19000 + i, 1.0 + i * 0.001) for i in range(30)] for k in keys}

        def _batch_historical(self, keys, days):
            raise HttpStatusError(400, "bad coin")

    fl = FakeLlama()
    got = fl.current(["good:1", "bad:2", "good:3"])
    check(got.get("good:1") == 1.5 and got.get("good:3") == 1.5 and "bad:2" not in got, "per-key retry after a rejected batch")
    keys = ["bad:a", "bad:b", "bad:c", "bad:d"] + [f"good:{i}" for i in range(30)]
    hist = fl.history(keys, 30)
    check(all(f"good:{i}" in hist for i in range(30)), "rejected keys do not trip the outage breaker")

    # --- spike in one marking print is capped; yield-bearing stable drifting up is not a de-peg
    spike = Pool("sp", "Base", "uniswap-v3", "A-B", 1e7, 5000.0, apy_base=5000.0, apy_base_7d=40.0, apy_mean_30d=40.0)
    check(close(marking_fee_apr(spike), 1.20), f"one-off fee print capped at 3x the 7-day level ({marking_fee_apr(spike)})")
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        y = Position(position_id="y", pool_id="y", chain="Base", project="aerodrome-v1", symbol="SUSDE-USDC",
                     opened_at=iso(T0), size_usd=1000.0, entry_apy=10.0, entry_fee_apr=0.1, entry_tvl=1e7,
                     entry_price_ratio=1.001, range_width=None, amp=1.0, fee_amp=1.0, sigma=0.01,
                     leg_symbols=["SUSDE", "USDC"], entry_prices=[1.001, 1.0], last_prices=[1.03, 1.0],
                     peg_legs=[0, 1], price_basis_at=iso(T0), drift_apr=0.12)
        check(mgr._peg_breach(y, T0 + 90 * 24 * H) is None, "yield-bearing stable accruing above $1 is not a de-peg")
        y.last_prices = [0.975, 1.0]
        check(mgr._peg_breach(y, T0 + 90 * 24 * H) is not None, "yield-bearing stable falling 2.5% is a de-peg")
        z = dataclasses_replace(y, symbol="USDE-USDC", leg_symbols=["USDE", "USDC"], entry_prices=[0.999, 1.0],
                                last_prices=[0.975, 1.0], drift_apr=0.0)
        check(mgr._peg_breach(z, T0 + 1 * H) is not None, "plain stable 2.5% under $1 is a de-peg")
        px.series["coingecko:tether"][-1] = 0.96
        res, why = mgr._entry_prices(pool("t", "Arbitrum", "uniswap-v3", "USDT-USDC", 14.0))
        check(res is None and "peg" in why, "a stablecoin already off its peg is not bought")

    # --- stableswap pegged pool has no range; CL pegged range widens with drift
    vol2 = VolEstimator(FixturePrices(series()), 90)
    cfg2 = Config()
    curve = evaluate(pool("c", "Arbitrum", "curve-dex", "USDT-USDC", 10.0, b7=10.0, mean=10.0), cfg2, vol2)
    check(curve.range_width is None, "stableswap pegged pool is not modelled as a range")
    from .scoring import stable_range_width
    check(stable_range_width(0.008, cfg2, 0.10) > stable_range_width(0.008, cfg2, 0.0) + 0.01,
          "ratio drift widens a pegged range so accrual does not walk it out")

    # --- holding decision ignores the exit cost
    hold = evaluate(pool("h", "Ethereum", "uniswap-v2", "WETH-USDC", 20.0, b7=20.0, mean=20.0), cfg2, vol2,
                    size_usd=1000.0, holding=True)
    check(close(hold.cost_drag_apr, 0.0), "holding economics carry no exit cost")

    # --- daily loss after a restart uses the first point after it
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        state.equity_history += [{"t": iso(T0 - 20 * 24 * H), "equity": 40000.0},
                                 {"t": iso(T0 - 5 * H), "equity": state.equity() * 1.3}]
        check(mgr._daily_loss_breached(T0), "the restart point is the reference on the restart day")


def test_review_round_three():
    from .prices import DefiLlamaPrices, HttpStatusError

    def legacy(pid, pool_id, chain, project, symbol, er, size=4000.0):
        return Position.from_dict({
            "position_id": pid, "pool_id": pool_id, "chain": chain, "project": project, "symbol": symbol,
            "opened_at": iso(T0 - 24 * H), "size_usd": size, "entry_apy": 20.0, "entry_fee_apr": 0.2,
            "entry_tvl": 30e6, "entry_price_ratio": er, "range_width": 3.0, "amp": 2.366, "fee_amp": 1.0,
            "sigma": 0.5, "value_usd": size, "last_price_ratio": er, "last_fee_apr": 0.2,
            "last_update": iso(T0 - 1 * H), "status": "open"})

    # stable-first legacy pair while prices are down: valued LP-vs-hold, never in quote units
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        eth = px.series["coingecko:ethereum"][-1]
        pos = legacy("sf", "usdc-weth", "Arbitrum", "uniswap-v3", "USDC-WETH", 1 / (eth / 1.14))
        pos.last_price_ratio = 1 / eth
        state.positions.append(pos)
        px.fail_current = True
        uni = [pool("usdc-weth", "Arbitrum", "uniswap-v3", "USDC-WETH", 20.0, b7=20.0, mean=20.0)]
        mgr.protect(uni, now=T0)
        check(abs(pos.value_usd - pos.size_usd) < 0.05 * pos.size_usd and close(pos.price_pnl_usd, 0.0),
              f"unconverted position keeps an LP-vs-hold value ({pos.value_usd:.2f})")
        rep = mgr.tick(uni, [], now=T0 + 72 * H)
        check(pos.status == "open", "no soft exit for a position that could not be priced this cycle")

    # legacy position whose pool is missing and whose tokens have no symbol price: missing grace governs
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        ms = legacy("ms", "msusd-mseth", "Base", "aerodrome-slipstream", "MSUSD-MSETH", 1 / 2400.0)
        state.positions.append(ms)
        others = mini_universe()
        mgr.protect(others, now=T0)
        mgr.protect(others, now=T0 + 7 * H)
        check(ms.status == "open" and not ms.leg_keys, "no 6h unpriced exit while the pool is missing; no keys guessed")
        mgr.protect(others, now=T0 + 25 * H)
        check(ms.status == "closed" and "missing" in (ms.close_reason or ""), "missing-pool grace closes it after 24h")

    # two conversions in different cycles still see a real drawdown
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr, px = make(tmp)
        a = legacy("a", "eth-arb", "Arbitrum", "uniswap-v3", "WETH-USDC", px.series["coingecko:ethereum"][-1])
        state.positions.append(a)
        state.cash_usd = ex._cash = 16_000.0
        uni = mini_universe()
        state.equity_history.append({"t": iso(T0 - 1 * H), "equity": state.equity()})
        mgr.protect(uni, now=T0)
        move(px, "coingecko:ethereum", 0.5)
        b = legacy("b", "sol-ray", "Solana", "raydium-amm", "SOL-USDC", px.series["coingecko:solana"][-1], size=1000.0)
        state.positions.append(b)
        state.cash_usd = ex._cash = state.cash_usd - 1000.0
        state.equity_history.append({"t": iso(T0 + 1 * H), "equity": state.equity()})
        mgr.protect(uni, now=T0 + 2 * H)
        check(mgr._daily_loss_breached(T0 + 2 * H), "a later conversion does not wipe out a real drawdown")

    # blanket 403 trips the breaker; same-id fallback is never used; marking cap; read retry
    class Blocked(DefiLlamaPrices):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def _get(self, path):
            self.calls += 1
            raise HttpStatusError(403, "blocked")

    bl = Blocked()
    bl.history([f"k:{i}" for i in range(200)], 30)
    check(bl.calls < 40, f"a blanket 403 is an outage, not 200 bad coins ({bl.calls} requests)")
    v = VolEstimator(FixturePrices(series()), 90)
    usdce = Pool("u", "Arbitrum", "uniswap-v3", "USDC.E-USDC", 1e7, 10.0, underlying_tokens=["0xff97", "0xaf88"])
    check(all(c[0] != c[1] for c in v.leg_candidates(usdce)), "a bridged token is never priced as its native twin")
    heavy = Pool("h", "Base", "aerodrome-slipstream", "A-B", 1e7, 240.0, apy_base=180.0, apy_reward=60.0,
                 apy_base_7d=4.0, apy_mean_30d=60.0)
    check(close(marking_fee_apr(heavy), 0.12), f"marking cap follows the 7-day fee, not reward-inflated means ({marking_fee_apr(heavy)})")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.json")
        save_state(path, State(cash_usd=1.0))
        save_state(path, State(cash_usd=2.0))
        import builtins
        real_open, n = builtins.open, {"c": 0}

        def flaky_open(f, *a, **k):
            if f == path and n["c"] < 2:
                n["c"] += 1
                raise PermissionError(13, "in use")
            return real_open(f, *a, **k)

        builtins.open = flaky_open
        try:
            st = load_state(path, 0.0)
        finally:
            builtins.open = real_open
        check(st.cash_usd == 2.0, "a briefly locked state file is retried, not rolled back to .bak")


def run_all() -> int:
    for name, fn in [("il", test_il), ("classification", test_classification),
                     ("price alignment+cache", test_price_alignment_and_cache),
                     ("sources+scoring", test_sources_and_scoring), ("entries+valuation", test_entries_and_valuation),
                     ("rewards+gaps", test_rewards_and_gaps), ("soft exits", test_soft_exits),
                     ("hard exits", test_hard_exits), ("rotation+loss limit", test_rotation_and_loss_limit),
                     ("legacy migration", test_legacy_migration), ("persistence+lock", test_persistence_and_lock),
                     ("review round two", test_review_round_two), ("review round three", test_review_round_three),
                     ("config", test_config)]:
        before = len(_fails)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            _fails.append(f"{name}: exception {exc!r}")
        print(f"{name:<24} {'ok' if len(_fails) == before else 'FAILED'}")
    if _fails:
        print(f"\n{len(_fails)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0
