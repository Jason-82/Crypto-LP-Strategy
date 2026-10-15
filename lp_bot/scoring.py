"""Filter and score pools into ranked opportunities.

net_apr = fee_amp * fee_apr + haircut * reward_apr - amp * sigma^2 / 8 - cost_drag

``cost_drag`` amortises gas, slippage and bridging over ``horizon_days`` at the size the bot
would actually buy. Every rejection carries a reason so ``scan --explain`` can show why a
tempting number was skipped. Nothing here places trades.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config import Config
from .il import amplification, breakeven_fee_apr, expected_il_rate
from .models import Opportunity, Pool
from .prices import VolEstimator, classify

CL_SLUG_TOKENS = {"v3", "v4", "clmm", "cl", "dlmm", "slipstream", "whirlpool", "whirlpools",
                  "concentrated", "elastic", "fusion", "algebra"}
CL_PROJECTS = {"orca", "orca-dex", "ekubo", "ambient", "maverick", "maverick-v2", "kamino-liquidity",
               "hyperion", "cetus", "cetus-amm", "cetus-clmm", "bluefin-spot", "turbos", "momentum",
               "byreal", "sparkdex-v4"}
CL_PROJECT_PREFIXES = ("trader-joe-v2", "lfj", "joe-v2", "kyberswap-elastic")
NON_CL_PREFIXES = ("curve", "balancer", "beethoven", "aura", "convex")
STABLESWAP_PREFIXES = ("curve", "balancer", "beethoven")


def is_concentrated(pool: Pool) -> bool:
    slug = pool.project.lower()
    if slug.startswith(NON_CL_PREFIXES):
        return False
    if slug in CL_PROJECTS or slug.startswith(CL_PROJECT_PREFIXES):
        return True
    return any(tok in CL_SLUG_TOKENS for tok in slug.split("-"))


def is_stableswap(pool: Pool) -> bool:
    return pool.project.lower().startswith(STABLESWAP_PREFIXES)


def stable_range_width(sigma: float, cfg: Config, drift: float = 0.0) -> float:
    """Range for a pegged pair in a concentrated pool: ``stable_range_sigmas`` horizon sigmas
    plus the ratio's own drift over two horizons (a yield-bearing leg walks away from the
    other), and at least 1 %."""
    years = max(cfg.horizon_days, 1) / 365.0
    half = cfg.stable_range_sigmas * sigma * math.sqrt(years) + abs(drift) * 2.0 * years
    return math.exp(max(half, 0.01))


def reject_reasons(pool: Pool, cfg: Config) -> List[str]:
    r: List[str] = []
    chain = pool.chain.lower()
    if cfg.chains and chain not in {c.lower() for c in cfg.chains}:
        r.append("chain not selected")
    if chain in {c.lower() for c in cfg.chain_blocklist}:
        r.append("chain blocked")
    if cfg.projects and pool.project.lower() not in {p.lower() for p in cfg.projects}:
        r.append("project not selected")
    if pool.project.lower() in {p.lower() for p in cfg.project_blocklist}:
        r.append("project blocked")
    blocked = {t.upper() for t in cfg.token_blocklist}
    if any(t in blocked or t.split(".")[0] in blocked for t in pool.tokens):
        r.append("token blocked")
    if cfg.require_multi_asset and (pool.exposure == "single" or len(pool.tokens) < 2):
        r.append("single-asset (not an LP pair)")
    if pool.tvl_usd < cfg.min_tvl_usd:
        r.append("tvl below floor")
    if pool.age_days is None:
        if not cfg.allow_unknown_age:
            r.append("pool age unknown")
    elif pool.age_days < cfg.min_pool_age_days:
        r.append("pool too young")
    if pool.apy < cfg.min_apy_pct:
        r.append("apy below floor")
    if pool.apy > cfg.max_apy_pct:
        r.append("apy implausibly high")
    if cfg.reject_outliers and pool.outlier:
        r.append("flagged outlier by source")
    if pool.apy > 0 and pool.apy_reward:
        share = pool.apy_reward / pool.apy
        if share > cfg.max_reward_share and not cfg.allow_reward_heavy:
            r.append("reward-emission heavy")
    if cfg.min_volume_tvl_ratio_7d > 0 and pool.volume_usd_7d is not None and pool.tvl_usd > 0:
        if pool.volume_usd_7d / pool.tvl_usd < cfg.min_volume_tvl_ratio_7d:
            r.append("turnover too low for fee apy to be real")
    return r


def _fee_ceiling(pool: Pool) -> float:
    """Fees can never exceed total APY minus emissions (percent)."""
    return max(pool.apy - (pool.apy_reward or 0.0), 0.0)


def conservative_fee_apr(pool: Pool, smooth: bool = False) -> Tuple[float, List[str]]:
    """Fee APR (decimal) for scoring.

    Default (entry): the lower of the 1-day and 7-day fee figures. ``smooth`` (holding): the
    7-day figure alone when available, so one quiet day cannot force a sale. Either way the
    result is capped at 1.5x the 30-day mean APY, halved when no 30-day mean exists, and never
    above total APY minus emissions."""
    notes: List[str] = []
    if smooth and pool.apy_base_7d is not None and pool.apy_base_7d >= 0:
        cands = [pool.apy_base_7d]
    else:
        cands = [x for x in (pool.apy_base, pool.apy_base_7d) if x is not None and x >= 0]
    if not cands:
        cands = [_fee_ceiling(pool)]
        notes.append("fee apy inferred as total minus reward")
    fee = min(cands)
    if not smooth:
        # Entry: never credit more than today's total minus emissions. (The smoothed holding
        # figure is a 7-day number and is not capped by a single day.)
        fee = min(fee, _fee_ceiling(pool))
    if pool.apy_mean_30d is not None and pool.apy_mean_30d >= 0:
        cap = 1.5 * pool.apy_mean_30d
        if fee > cap:
            notes.append(f"fee apy capped at 1.5x 30d mean ({pool.apy_mean_30d:.1f}%)")
            fee = cap
    else:
        notes.append("no 30d mean to cap against; fee apy halved")
        fee *= 0.5
    return fee / 100.0, notes


MAX_MARK_APY_PCT = 250.0


def marking_fee_apr(pool: Pool) -> float:
    """Best estimate of what the pool is paying right now (decimal), used to accrue paper fees.
    No haircuts, but a one-hour print cannot exceed 3x the 7-day or 30-day level (or 250 %),
    so a bad data point cannot book phantom fees that would mask a real IL loss."""
    fee = None
    for x in (pool.apy_base, pool.apy_base_7d):
        if x is not None and x >= 0:
            fee = min(x, _fee_ceiling(pool))
            break
    if fee is None:
        fee = _fee_ceiling(pool)
    if pool.apy_base_7d is not None and pool.apy_base_7d >= 0:
        cap = 3.0 * pool.apy_base_7d
    elif pool.apy_mean_30d is not None and pool.apy_mean_30d >= 0:
        cap = 3.0 * max(pool.apy_mean_30d - (pool.apy_reward or 0.0), 0.0)  # 30d mean is total APY
    else:
        cap = MAX_MARK_APY_PCT
    return min(fee, cap, MAX_MARK_APY_PCT) / 100.0


def ranking_size(cfg: Config, pool: Pool, equity: Optional[float]) -> float:
    """The size the bot would actually buy, so costs are amortised realistically."""
    size = cfg.max_position_usd
    if equity:
        size = min(size, cfg.max_pct_per_pool * equity)
    if pool.tvl_usd > 0:
        size = min(size, cfg.max_share_of_pool_tvl * pool.tvl_usd)
    return max(size, cfg.min_position_usd)


def evaluate(pool: Pool, cfg: Config, vol: VolEstimator, size_usd: Optional[float] = None,
             holding: bool = False, equity: Optional[float] = None) -> Opportunity:
    """Score a pool. ``holding=True`` scores an existing position: smoothed fees and no costs
    (the entry cost is spent and the exit cost is paid whether we sell now or later)."""
    fee_apr, notes = conservative_fee_apr(pool, smooth=holding)
    reward_apr = (pool.apy_reward or 0.0) / 100.0 * cfg.reward_haircut
    sigma, vol_source = vol.estimate(pool)
    if pool.stablecoin and pool.il_risk == "no" and vol_source != "history":
        sigma = min(sigma, 0.02)
    stable_pair = sigma < cfg.stable_sigma_threshold
    concentrated = is_concentrated(pool)
    drift = vol.drift(pool) if stable_pair else 0.0
    if stable_pair and concentrated:
        # Pegged pairs are LP'd tightly around the peg; the pool APY reflects that, and so must
        # the IL and out-of-range model.
        range_width: Optional[float] = stable_range_width(sigma, cfg, drift)
        fee_amp = 1.0
        notes.append(f"stable pair: modelled as a +/-{(range_width - 1) * 100:.1f}% range")
    elif concentrated:
        range_width = cfg.default_range_width
        fee_amp = 1.0 + (amplification(range_width) - 1.0) * cfg.cl_fee_amp_share
    else:
        range_width = None
        fee_amp = 1.0
    amp = amplification(range_width)
    exp_il = expected_il_rate(sigma, amp)

    size = size_usd or ranking_size(cfg, pool, equity)
    cost = 0.0 if holding else 2.0 * cfg.one_way_cost(pool.chain, size, stable_pair)
    cost_drag = (cost / size) * (365.0 / max(cfg.horizon_days, 1))

    net = fee_amp * fee_apr + reward_apr - exp_il - cost_drag

    conf = 1.0
    if pool.age_days is None or pool.age_days < 90:
        conf *= 0.85
    if pool.tvl_usd < 10_000_000:
        conf *= 0.9
    if not vol_source.startswith("history"):
        conf *= 0.9
    if pool.apy > 0 and (pool.apy_reward or 0.0) / pool.apy > 0.3:
        conf *= 0.9
    if any(classify(t) == "exotic" for t in pool.tokens):
        conf *= 0.8
        notes.append("contains an unclassified token")

    return Opportunity(
        pool=pool, fee_apr=fee_apr, reward_apr=reward_apr, sigma=sigma, vol_source=vol_source,
        amp=amp, fee_amp=fee_amp, range_width=range_width, expected_il_apr=exp_il,
        cost_drag_apr=cost_drag, net_apr=net,
        breakeven_fee_apr=breakeven_fee_apr(sigma, amp, fee_amp),
        confidence=conf, score=net * conf, notes=notes, size_usd=size, stable_pair=stable_pair,
        drift=drift,
    )


def rank(pools: List[Pool], cfg: Config, vol: VolEstimator, equity: Optional[float] = None,
         budget_s: Optional[float] = None) -> Tuple[List[Opportunity], Counter, Dict[str, List[str]]]:
    """Return (ranked opportunities, rejection tally, {pool_id: reasons})."""
    tally: Counter = Counter()
    why: Dict[str, List[str]] = {}
    survivors: List[Pool] = []
    for p in pools:
        reasons = reject_reasons(p, cfg)
        if reasons:
            tally.update(reasons)
            why[p.pool_id] = reasons
        else:
            survivors.append(p)
    vol.reset_stats()
    vol.prefetch(survivors, budget_s=budget_s)
    opps: List[Opportunity] = []
    for p in survivors:
        o = evaluate(p, cfg, vol, equity=equity)
        post: List[str] = []
        if o.sigma > cfg.max_sigma:
            post.append("pair too volatile")
        if o.expected_il_apr > cfg.max_expected_il_apr:
            post.append("expected IL too large")
        if o.net_apr < cfg.min_net_apr:
            post.append("net apr below floor after IL and costs")
        if post:
            tally.update(post)
            why[p.pool_id] = post
        else:
            opps.append(o)
    opps.sort(key=lambda o: o.score, reverse=True)
    return opps, tally, why
