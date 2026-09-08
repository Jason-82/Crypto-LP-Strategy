"""Filter and score pools into ranked opportunities.

net_apr = fee_amp * fee_apr + haircut * reward_apr - amp * sigma^2 / 8 - cost_drag

Every rejection carries a reason so ``scan --explain`` can show why a tempting number was
skipped. Nothing here places trades.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config import Config
from .il import amplification, breakeven_fee_apr, expected_il_rate
from .models import Opportunity, Pool
from .prices import VolEstimator, classify

CL_SLUG_TOKENS = {"v3", "v4", "clmm", "cl", "dlmm", "slipstream", "whirlpool", "concentrated"}
CL_PROJECTS = {"orca", "ekubo", "ambient", "maverick", "maverick-v2", "meteora", "kamino-liquidity"}


def is_concentrated(pool: Pool) -> bool:
    slug = pool.project.lower()
    if slug in CL_PROJECTS:
        return True
    return any(tok in CL_SLUG_TOKENS for tok in slug.split("-"))


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
    if any(t in blocked for t in pool.tokens):
        r.append("token blocked")
    if cfg.require_multi_asset and (pool.exposure == "single" or len(pool.tokens) < 2):
        r.append("single-asset (not an LP pair)")
    if pool.tvl_usd < cfg.min_tvl_usd:
        r.append("tvl below floor")
    if pool.age_days is not None and pool.age_days < cfg.min_pool_age_days:
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


def conservative_fee_apr(pool: Pool) -> Tuple[float, List[str]]:
    """Fee APR (decimal) that leans on the slower-moving numbers when they exist."""
    notes: List[str] = []
    cands = [x for x in (pool.apy_base, pool.apy_base_7d) if x is not None and x >= 0]
    if not cands:
        base = max(pool.apy - (pool.apy_reward or 0.0), 0.0)
        cands = [base]
        notes.append("fee apy inferred as total minus reward")
    fee = min(cands)
    if pool.apy_mean_30d is not None and pool.apy_mean_30d > 0:
        cap = 1.5 * pool.apy_mean_30d
        if fee > cap:
            notes.append(f"fee apy capped at 1.5x 30d mean ({pool.apy_mean_30d:.1f}%)")
            fee = cap
    return fee / 100.0, notes


def evaluate(pool: Pool, cfg: Config, vol: VolEstimator, size_usd: Optional[float] = None) -> Opportunity:
    fee_apr, notes = conservative_fee_apr(pool)
    reward_apr = (pool.apy_reward or 0.0) / 100.0 * cfg.reward_haircut
    sigma, vol_source, _ratio = vol.estimate(pool)
    if pool.stablecoin and pool.il_risk == "no" and vol_source != "history":
        sigma = min(sigma, 0.02)
    concentrated = is_concentrated(pool)
    range_width = cfg.default_range_width if concentrated else None
    if concentrated and sigma < cfg.stable_sigma_threshold:
        # Stable pairs are LP'd in tight ranges; the pool-level APY already reflects that and
        # the symmetric wide-range model does not apply. Take the pool number at face value.
        range_width = None
        notes.append("stable pair: pool APY taken at face value, no range model")
    amp = amplification(range_width)
    fee_amp = 1.0 + (amp - 1.0) * cfg.cl_fee_amp_share
    exp_il = expected_il_rate(sigma, amp)

    size = size_usd or cfg.max_position_usd
    roundtrip = cfg.gas_for(pool.chain) + 2.0 * size * cfg.slippage_bps_per_side / 10_000.0
    cost_drag = (roundtrip / size) * (365.0 / max(cfg.horizon_days, 1))

    net = fee_amp * fee_apr + reward_apr - exp_il - cost_drag

    conf = 1.0
    if pool.age_days is not None and pool.age_days < 90:
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
        confidence=conf, score=net * conf, notes=notes,
    )


def rank(pools: List[Pool], cfg: Config, vol: VolEstimator) -> Tuple[List[Opportunity], Counter, Dict[str, List[str]]]:
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
    vol.prefetch(survivors)
    opps: List[Opportunity] = []
    for p in survivors:
        o = evaluate(p, cfg, vol)
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
