"""Deterministic checks: IL arithmetic, volatility estimation, scoring filters, paper fills,
and the manager's exit/rotation/kill-switch behaviour on synthetic data. No network."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from datetime import datetime, timedelta, timezone

from .config import Config, load_config
from .executors import PaperExecutor
from .il import (amplification, expected_il_rate, il_constant_product, impermanent_loss, in_range,
                 pair_sigma, position_value, price_move_for_il)
from .manager import Manager, iso
from .models import Pool
from .prices import (FixturePrices, NullPrices, VolEstimator, classify, heuristic_sigma,
                     realised_vol, ratio_sigma_from_history)
from .scoring import conservative_fee_apr, is_concentrated, rank, reject_reasons
from .sources import FilePoolSource, parse_llama_pool
from .state import State, load_state, save_state

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "pools_sample.json")
T0 = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)

_fails = []


def check(cond, msg):
    if not cond:
        _fails.append(msg)
        print(f"  FAIL  {msg}")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_il():
    check(close(il_constant_product(4.0), -0.2), "IL at 4x should be -20%")
    check(close(il_constant_product(1.0), 0.0), "IL at 1x is zero")
    check(close(il_constant_product(2.0), il_constant_product(0.5)), "IL symmetric in log space")
    check(close(amplification(4.0), 2.0), "amp for [P/4, 4P] is 2")
    check(close(amplification(None), 1.0), "full range amp is 1")
    a = amplification(2.0)
    check(close(impermanent_loss(1.5, 2.0), a * il_constant_product(1.5)), "in-range CL IL = amp x v2 IL")
    # continuity at the range boundary and monotone decline beyond it
    check(close(position_value(2.0, 2.0), position_value(2.0000001, 2.0), 1e-6), "value continuous at top")
    check(close(position_value(0.5, 2.0), position_value(0.4999999, 2.0), 1e-6), "value continuous at bottom")
    check(impermanent_loss(3.0, 2.0) < impermanent_loss(2.0, 2.0), "IL keeps growing above range")
    check(impermanent_loss(0.25, 2.0) < impermanent_loss(0.5, 2.0), "IL keeps growing below range")
    check(close(impermanent_loss(3.0, None), il_constant_product(3.0)), "full range == v2")
    check(in_range(1.9, 2.0) and not in_range(2.1, 2.0) and in_range(9.0, None), "in_range")
    check(close(expected_il_rate(0.7, 1.0), 0.06125), "E[IL] = sigma^2/8")
    check(close(price_move_for_il(-0.2, None), 4.0, 1e-6), "20% IL budget is a 4x move full range")
    k = price_move_for_il(-0.03, 2.0)
    check(close(impermanent_loss(k, 2.0), -0.03, 1e-6), "IL budget inversion for CL")
    check(close(pair_sigma(0.6, 0.6, 1.0), 0.0) and close(pair_sigma(0.6, 0.0, 0.0), 0.6), "pair sigma")


def test_prices():
    check(classify("USDC") == "stable" and classify("wstETH") == "eth" and classify("cbBTC") == "btc", "classify")
    check(classify("SOL") == "major" and classify("BRETT") == "exotic", "classify majors/exotic")
    check(close(heuristic_sigma(["USDC", "USDT"])[0], 0.02), "stable pair heuristic")
    check(classify("EURC") == "eurstable" and close(heuristic_sigma(["EURC", "USDC"])[0], 0.08), "euro stable vs usd stable carries FX vol")
    check(close(heuristic_sigma(["EURC", "EURS"])[0], 0.02), "euro/euro is a stable pair")
    check(close(heuristic_sigma(["WETH", "USDC"])[0], 0.70), "eth/stable heuristic")
    check(close(heuristic_sigma(["DAI", "USDC", "USDT"])[0], 0.02), "3pool heuristic")
    check(heuristic_sigma(["BRETT", "WETH"])[0] > 1.0, "exotic heuristic")
    check(close(heuristic_sigma(["USDC"])[0], 0.0), "single asset heuristic")

    rng = random.Random(7)
    sigma_d = 0.8 / math.sqrt(365)
    eth, usd = [2000.0], [1.0]
    for _ in range(365 * 4):
        eth.append(eth[-1] * math.exp(rng.gauss(0, sigma_d)))
        usd.append(1.0)
    rv = realised_vol(eth)
    check(rv is not None and 0.7 < rv < 0.9, f"realised vol recovers ~0.8 (got {rv})")
    rs = ratio_sigma_from_history(eth, usd)
    check(rs is not None and close(rs, rv, 1e-9), "ratio sigma vs flat leg equals leg vol")
    check(realised_vol([1.0] * 5) is None, "too-short history -> None")

    fx = FixturePrices({"coingecko:ethereum": eth[-90:], "coingecko:usd-coin": usd[-90:]})
    vol = VolEstimator(fx, 90)
    pool = parse_llama_pool({"pool": "x", "chain": "Ethereum", "project": "uniswap-v3", "symbol": "WETH-USDC",
                             "tvlUsd": 1e7, "apy": 10})
    sig, src, ratio = vol.estimate(pool)
    check(src == "history" and 0.4 < sig < 1.4, f"history-based sigma used ({src}, {sig:.2f})")
    check(close(ratio, eth[-1]), "current ratio from history")
    check(close(vol.current_ratio(pool), eth[-1]), "current_ratio from price source")
    pool2 = parse_llama_pool({"pool": "y", "chain": "Base", "project": "uniswap-v3", "symbol": "BRETT-WETH",
                              "tvlUsd": 1e7, "apy": 10})
    sig2, src2, _ = VolEstimator(NullPrices(), 90).estimate(pool2)
    check(src2.startswith("heuristic") and sig2 > 1.0, "falls back to heuristic without history")


def test_sources_and_scoring():
    pools = FilePoolSource(FIXTURE).fetch()
    check(len(pools) == 20, f"fixture parses 20 pools (got {len(pools)})")
    p = next(x for x in pools if x.pool_id == "fx-eth-uni3-usdc-weth")
    check(p.tokens == ["USDC", "WETH"] and p.age_days == 900 and p.apy_base_7d == 12.8, "llama parse fields")
    check(is_concentrated(p) and not is_concentrated(next(x for x in pools if x.project == "curve-dex")),
          "concentrated detection")
    check(is_concentrated(Pool("a", "Solana", "orca", "SOL-USDC", 1, 1)), "orca is concentrated")
    check(not is_concentrated(Pool("a", "Ethereum", "clearpool", "USDC", 1, 1)), "clearpool is not CL")
    fee, notes = conservative_fee_apr(p)
    check(close(fee, 0.128), "conservative fee apr takes the 7d figure")
    capped = Pool("c", "Ethereum", "uniswap-v3", "A-B", 1e7, 100.0, apy_base=100.0, apy_mean_30d=20.0)
    check(close(conservative_fee_apr(capped)[0], 0.30), "fee apr capped at 1.5x 30d mean")
    no30 = Pool("n", "Ethereum", "uniswap-v3", "A-B", 1e7, 100.0, apy_base=100.0)
    check(close(conservative_fee_apr(no30)[0], 0.50), "fee apr halved without a 30d mean")

    cfg = Config()
    stable_cl = Pool("s", "Ethereum", "ekubo", "USDC-USDG", 5e6, 20.0, apy_base=20.0)
    from .scoring import evaluate
    o = evaluate(stable_cl, cfg, VolEstimator(NullPrices(), 90))
    check(o.range_width is None and close(o.amp, 1.0), "stable pair in a CL pool skips the range model")
    vol_cl = evaluate(Pool("v", "Base", "uniswap-v3", "WETH-USDC", 5e6, 30.0, apy_base=30.0), cfg,
                      VolEstimator(NullPrices(), 90))
    check(vol_cl.range_width == cfg.default_range_width and vol_cl.amp > 2 and close(vol_cl.fee_amp, 1.0),
          "volatile CL pair: amplified IL, pool-average fees")
    opps, tally, why = rank(pools, cfg, VolEstimator(NullPrices(), 90))
    ids = {o.pool.pool_id for o in opps}
    expect_reject = {
        "fx-eth-scam-farm": "apy implausibly high",
        "fx-eth-aave-usdc": "single-asset (not an LP pair)",
        "fx-eth-curve-3pool": "apy below floor",
        "fx-arb-camelot3-grail-weth": "reward-emission heavy",
        "fx-base-uni3-usdc-cbbtc": "pool too young",
        "fx-base-aero-usdc-aero": "reward-emission heavy",
        "fx-base-uni3-brett-weth": "expected IL too large",
        "fx-pol-uni3-wmatic-usdc": "net apr below floor after IL and costs",
    }
    for pid, reason in expect_reject.items():
        check(pid not in ids and reason in why.get(pid, []), f"{pid} rejected for '{reason}' (got {why.get(pid)})")
    for pid in ("fx-op-velo3-wsteth-weth", "fx-base-aero-slip-weth-usdc", "fx-arb-uni3-weth-usdc",
                "fx-sol-orca-sol-usdc", "fx-sol-ray-clmm-sol-usdc", "fx-sui-cetus-sui-usdc"):
        check(pid in ids, f"{pid} survives filters ({why.get(pid)})")
    # mainnet gas at a $10k size is a real drag: the Ethereum stable pools do not clear costs
    for pid in ("fx-eth-curve-usde-usdc", "fx-eth-curve-steth-eth"):
        check("net apr below floor after IL and costs" in why.get(pid, []), f"{pid} killed by mainnet costs")
    cheap = Config(gas_roundtrip_usd={"ethereum": 2.0, "default": 2.0}, slippage_bps_per_side=2.0, horizon_days=90)
    opps_cg, _, _ = rank(pools, cheap, VolEstimator(NullPrices(), 90))
    check("fx-eth-curve-usde-usdc" in {o.pool.pool_id for o in opps_cg},
          "same stable pool survives with cheap execution and a longer horizon")
    scores = [o.score for o in opps]
    check(scores == sorted(scores, reverse=True), "ranked by score desc")
    check(all(o.net_apr >= cfg.min_net_apr for o in opps), "all survivors clear net floor")
    check(sum(tally.values()) >= len(expect_reject), "tally populated")

    cfg2 = Config(chains=["solana"])
    opps2, _, _ = rank(pools, cfg2, VolEstimator(NullPrices(), 90))
    check(opps2 and all(o.pool.chain == "Solana" for o in opps2), "chain filter")
    cfg3 = Config(token_blocklist=["SOL"])
    check("token blocked" in reject_reasons(next(x for x in pools if x.symbol == "SOL-USDC"), cfg3), "token blocklist")


def test_paper_executor():
    cfg = Config(slippage_bps_per_side=30, gas_roundtrip_usd={"arbitrum": 1.5, "default": 3.0})
    ex = PaperExecutor(cfg, 25_000.0)
    pools = FilePoolSource(FIXTURE).fetch()
    opps, _, _ = rank(pools, cfg, VolEstimator(NullPrices(), 90))
    arb = next(o for o in opps if o.pool.chain == "Arbitrum")
    fill = ex.open(arb, 1000.0)
    check(fill.ok and close(fill.cost_usd, 0.75 + 3.0) and close(fill.size_usd, 996.25), "paper open cost")
    check(close(ex.cash_usd(), 24_000.0), "paper open debits cash")
    check(not ex.open(arb, 50_000.0).ok, "paper open rejects oversize")


def _mini_pools():
    raw = json.load(open(FIXTURE))["data"]
    keep = {"fx-base-aero-slip-weth-usdc", "fx-op-velo3-wsteth-weth", "fx-sol-orca-sol-usdc", "fx-sol-ray-clmm-sol-usdc"}
    return [parse_llama_pool(d) for d in raw if d["pool"] in keep]


def _mgr(tmp, **over):
    cfg = Config(state_path=os.path.join(tmp, "state.json"), kill_switch_path=os.path.join(tmp, "STOP"),
                 starting_cash_usd=20_000.0, min_hold_hours=48, **over)
    state = load_state(cfg.state_path, cfg.starting_cash_usd)
    ex = PaperExecutor(cfg, state.cash_usd)
    vol = VolEstimator(NullPrices(), 90)
    return cfg, state, ex, vol, Manager(cfg, state, ex, vol)


def test_manager():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, state, ex, vol, mgr = _mgr(tmp)
        pools = _mini_pools()
        opps, _, _ = rank(pools, cfg, vol)
        check(len(opps) == 4, f"mini universe has 4 candidates (got {len(opps)})")
        rep = mgr.tick(pools, opps, now=T0, ratios={})
        n = len(state.open_positions)
        check(n == 4, f"entered 4 positions (got {n})")
        equity = state.equity()
        check(all(p.size_usd <= cfg.max_pct_per_pool * 20_000 + 1e-6 for p in state.open_positions), "per-pool cap")
        tiny = Config(state_path=os.path.join(tmp, "s2.json"), kill_switch_path=os.path.join(tmp, "K"),
                      max_share_of_pool_tvl=0.0001)
        st2 = State(cash_usd=20_000.0)
        m2 = Manager(tiny, st2, PaperExecutor(tiny, 20_000.0), vol)
        acts = m2.plan(opps, {p.pool_id: p for p in pools}, T0)
        check(all(a.size_usd <= 0.0001 * a.opportunity.pool.tvl_usd + 1e-6 for a in acts if a.kind == "enter"),
              "position capped as a share of pool tvl")
        check(state.cash_usd >= cfg.cash_reserve_pct * equity - 1e-6, "cash reserve kept")
        check(19_900 < equity < 20_000, f"equity after entry costs ~20k (got {equity:.2f})")
        save_state(cfg.state_path, state)
        reloaded = load_state(cfg.state_path, 0.0)
        check(len(reloaded.open_positions) == 4 and close(reloaded.cash_usd, state.cash_usd), "state round-trip")

        # fees accrue on the correlated wstETH/WETH pool held 30 days at flat prices
        velo = next(p for p in state.open_positions if p.pool_id == "fx-op-velo3-wsteth-weth")
        velo_pool = next(p for p in pools if p.pool_id == velo.pool_id)
        t30 = T0 + timedelta(days=30)
        mgr.mark(velo, velo_pool, 1.0, t30)
        expect = velo.size_usd * velo.last_fee_apr * velo.fee_amp * 30 / 365
        check(close(velo.fees_accrued_usd, expect, 1e-6) and close(velo.il_usd, 0.0), "fee accrual at flat prices")
        velo.fees_accrued_usd, velo.last_update = 0.0, iso(T0)

        # a 1.6x move on the concentrated SOL position breaches the 3% IL budget -> hard exit
        sol = next(p for p in state.open_positions if p.pool_id == "fx-sol-orca-sol-usdc")
        t1 = T0 + timedelta(hours=1)
        rep = mgr.tick(pools, opps, now=t1, ratios={sol.pool_id: 1.6})
        exits = [a for a in rep["actions"] if a["kind"] == "exit"]
        check(len(exits) == 1 and "IL budget" in exits[0]["note"], f"IL stop fired ({exits})")
        check(sol.status == "closed" and sol.realised_pnl_usd < 0, "SOL position closed at a loss")
        check(len(state.open_positions) == 3 and sol.pool_id not in {p.pool_id for p in state.open_positions},
              "no immediate re-entry into a pool that just stopped out (cooldown)")

        # soft exit (yield decay) waits for min hold, then fires
        wb = next(p for p in state.open_positions if p.pool_id == "fx-sol-ray-clmm-sol-usdc")
        decayed = [p for p in pools if p.pool_id != wb.pool_id]
        wb_pool = next(p for p in pools if p.pool_id == wb.pool_id)
        decayed_pool = Pool(**{**wb_pool.to_dict(), "apy": 1.0, "apy_base": 1.0, "apy_base_7d": 1.0})
        decayed.append(decayed_pool)
        r_early = mgr.exit_reason(wb, decayed_pool, T0 + timedelta(hours=2))
        check(r_early is None, "soft exit suppressed inside min hold")
        r_late = mgr.exit_reason(wb, decayed_pool, T0 + timedelta(hours=72))
        check(r_late is not None and "yield decayed" in r_late[0] and not r_late[1], "yield decay exit after min hold")

        # TVL drain
        drained = Pool(**{**wb_pool.to_dict(), "tvl_usd": wb.entry_tvl * 0.3})
        r = mgr.exit_reason(wb, drained, T0 + timedelta(hours=72))
        check(r is not None and "TVL fell" in r[0], "tvl drain exit")

        # out-of-range handling for concentrated positions (range width 3 -> 3.5x is outside)
        velo = next(p for p in state.open_positions if p.pool_id == "fx-base-aero-slip-weth-usdc")
        velo_pool = next(p for p in pools if p.pool_id == velo.pool_id)
        mgr.mark(velo, velo_pool, 3.5, T0 + timedelta(hours=3))
        check(velo.out_of_range_since is not None, "out-of-range timestamp set")
        r = mgr.exit_reason(velo, velo_pool, T0 + timedelta(hours=3))
        check(r is not None and "IL budget" in r[0], "big move trips IL stop before grace")
        # reset to a small, in-range move and confirm no exit
        mgr.mark(velo, velo_pool, 1.01, T0 + timedelta(hours=4))
        check(velo.out_of_range_since is None and mgr.exit_reason(velo, velo_pool, T0 + timedelta(hours=4)) is None,
              "back in range clears the clock")

        # pool disappears -> hard exit
        r = mgr.exit_reason(velo, None, T0 + timedelta(hours=4))
        check(r is not None and r[1], "missing pool is a hard exit")

        # kill switch -> everything exits, nothing enters
        open(cfg.kill_switch_path, "w").close()
        rep = mgr.tick(pools, opps, now=T0 + timedelta(hours=5), ratios={})
        check(len(state.open_positions) == 0, "kill switch closed everything")
        check(not any(a["kind"] == "enter" for a in rep["actions"]), "kill switch blocks entries")
        os.unlink(cfg.kill_switch_path)
        check(state.cash_usd > 19_000, f"cash returned after exits ({state.cash_usd:.2f})")


def test_rotation_and_loss_limit():
    with tempfile.TemporaryDirectory() as tmp:
        # a 90-day horizon so the +12 pp gain outweighs two rounds of switching costs
        cfg, state, ex, vol, mgr = _mgr(tmp, max_positions=1, rotation_margin_apr=0.05, horizon_days=90)
        pools = _mini_pools()
        weak = next(p for p in FilePoolSource(FIXTURE).fetch() if p.pool_id == "fx-arb-uni3-weth-usdc")
        strong = next(p for p in pools if p.pool_id == "fx-sol-ray-clmm-sol-usdc")
        opps_weak, _, _ = rank([weak], cfg, vol)
        mgr.tick([weak], opps_weak, now=T0, ratios={})
        check(len(state.open_positions) == 1, "single position opened for rotation test")
        opps_both, _, _ = rank([weak, strong], cfg, vol)
        check(opps_both[0].pool.pool_id == strong.pool_id, "stronger candidate ranks first")
        rep = mgr.tick([weak, strong], opps_both, now=T0 + timedelta(hours=1), ratios={})
        check(not any(a["kind"] == "exit" for a in rep["actions"]), "no rotation inside min hold")
        rep = mgr.tick([weak, strong], opps_both, now=T0 + timedelta(hours=72), ratios={})
        kinds = [(a["kind"], a["note"]) for a in rep["actions"]]
        check(any(k == "exit" and n.startswith("rotate") for k, n in kinds), f"rotation exit ({kinds})")
        check(any(k == "enter" for k, n in kinds) and state.open_positions[0].pool_id == strong.pool_id,
              f"rotated into the stronger pool ({kinds})")

        # daily loss limit blocks new entries
        state.equity_history.append({"t": iso(T0 + timedelta(hours=72)), "equity": state.equity() * 1.2})
        cfg.max_positions = 3
        rep = mgr.tick([weak, strong], opps_both, now=T0 + timedelta(hours=100), ratios={})
        check(any(a["kind"] == "skip" and "daily loss" in a["note"] for a in rep["actions"]),
              "daily loss limit blocks entries")


def test_config():
    cfg = load_config(os.path.join(HERE, "config.example.json"))
    check(cfg.mode == "paper" and cfg.gas_for("Ethereum") == 60 and cfg.gas_for("zksync") == 3.0, "example config loads")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"bogus": 1}, fh)
    try:
        load_config(fh.name)
        check(False, "unknown key should raise")
    except ValueError:
        pass
    finally:
        os.unlink(fh.name)


def run_all() -> int:
    for name, fn in [("il", test_il), ("prices", test_prices), ("sources+scoring", test_sources_and_scoring),
                     ("paper executor", test_paper_executor), ("manager", test_manager),
                     ("rotation+loss limit", test_rotation_and_loss_limit), ("config", test_config)]:
        before = len(_fails)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            _fails.append(f"{name}: exception {exc!r}")
        status = "ok" if len(_fails) == before else "FAILED"
        print(f"{name:<22} {status}")
    if _fails:
        print(f"\n{len(_fails)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0
