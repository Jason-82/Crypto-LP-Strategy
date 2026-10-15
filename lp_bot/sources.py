"""Pool sources. DefiLlama's yields API aggregates thousands of pools across every major
chain and DEX, which is why it is the default venue scanner. Add further sources by
implementing ``PoolSource.fetch`` and returning ``Pool`` records; the scorer does not care
where a pool came from.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

from .models import Pool


def _f(v) -> Optional[float]:
    try:
        x = None if v is None else float(v)
    except (TypeError, ValueError):
        return None
    if x is not None and x != x:  # NaN
        return None
    return x


def _i(v) -> Optional[int]:
    x = _f(v)
    return None if x is None else int(x)


def parse_llama_pool(d: Dict) -> Pool:
    """Convert one record from https://yields.llama.fi/pools into a Pool."""
    apy = _f(d.get("apy"))
    apy_base = _f(d.get("apyBase"))
    apy_reward = _f(d.get("apyReward"))
    if apy is None:
        apy = (apy_base or 0.0) + (apy_reward or 0.0)
    if apy_base is None and apy_reward is not None:
        apy_base = max(apy - apy_reward, 0.0)
    return Pool(
        pool_id=str(d.get("pool")),
        chain=str(d.get("chain", "")),
        project=str(d.get("project", "")),
        symbol=str(d.get("symbol", "")),
        tvl_usd=_f(d.get("tvlUsd")) or 0.0,
        apy=apy,
        apy_base=apy_base,
        apy_reward=apy_reward,
        apy_base_7d=_f(d.get("apyBase7d")),
        apy_mean_30d=_f(d.get("apyMean30d")),
        stablecoin=bool(d.get("stablecoin", False)),
        il_risk=str(d.get("ilRisk", "yes")),
        exposure=str(d.get("exposure", "multi")),
        outlier=bool(d.get("outlier", False)),
        age_days=_i(d.get("count")),
        underlying_tokens=[str(t) for t in (d.get("underlyingTokens") or []) if t],
        volume_usd_1d=_f(d.get("volumeUsd1d")),
        volume_usd_7d=_f(d.get("volumeUsd7d")),
        pool_meta=d.get("poolMeta"),
        url=f"https://defillama.com/yields/pool/{d.get('pool')}",
    )


def parse_records(data) -> Tuple[List[Pool], int]:
    """Parse records one at a time; a malformed record is skipped, not fatal."""
    pools: List[Pool] = []
    skipped = 0
    for d in data or []:
        try:
            if not isinstance(d, dict) or not d.get("pool"):
                raise ValueError("no pool id")
            pools.append(parse_llama_pool(d))
        except (TypeError, ValueError, KeyError, AttributeError):
            skipped += 1
    return pools, skipped


class PoolSource:
    name = "base"
    skipped = 0      # malformed records dropped by the last fetch

    def fetch(self) -> List[Pool]:  # pragma: no cover - interface
        raise NotImplementedError


class DefiLlamaYields(PoolSource):
    """Live scan of every pool DefiLlama tracks (all chains, all protocols it indexes)."""

    name = "defillama"

    def __init__(self, url: str = "https://yields.llama.fi/pools", timeout_s: int = 30):
        self.url = url
        self.timeout_s = timeout_s

    def fetch(self) -> List[Pool]:
        req = urllib.request.Request(self.url, headers={"User-Agent": "lp_bot/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.load(resp)
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        pools, self.skipped = parse_records(data)
        return pools


class FilePoolSource(PoolSource):
    """Pools from a JSON file (raw DefiLlama response or a bare list). Used for testing,
    offline runs, and replaying a saved snapshot."""

    name = "file"

    def __init__(self, path: str):
        self.path = path

    def fetch(self) -> List[Pool]:
        with open(self.path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        pools, self.skipped = parse_records(data)
        return pools


class MultiSource(PoolSource):
    """Concatenate several sources, de-duplicating on pool_id (first wins)."""

    name = "multi"

    def __init__(self, sources: Iterable[PoolSource]):
        self.sources = list(sources)

    def fetch(self) -> List[Pool]:
        seen, out = set(), []
        for src in self.sources:
            for p in src.fetch():
                if p.pool_id not in seen:
                    seen.add(p.pool_id)
                    out.append(p)
        return out
