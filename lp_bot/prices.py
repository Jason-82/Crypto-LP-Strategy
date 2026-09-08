"""Token prices and pair-volatility estimation.

Volatility of the price *ratio* of a pool's two legs is the single input that matters most
for impermanent loss. We try, in order:

1. Realised volatility from daily price history (DefiLlama coins API, any chain).
2. A heuristic by token class (stable / ETH-family / BTC-family / major / exotic).

Every estimate carries its provenance so the report can say which one was used.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .il import pair_sigma
from .models import Pool

STABLES = {
    "USDC", "USDT", "DAI", "USDE", "SUSDE", "FRAX", "LUSD", "GHO", "CRVUSD", "USDS", "SUSDS",
    "PYUSD", "SUSD", "TUSD", "USDD", "MIM", "FDUSD", "USD0", "USDC.E", "USDT.E", "USDBC",
    "USDB", "DOLA", "BOLD", "GUSD", "USDP", "EURC", "EURS", "AGEUR", "EURA", "USDM", "USDY",
    "USDA", "USDX", "USR", "RLUSD", "DEUSD", "SFRAX", "SDAI", "USD+", "USDT0", "BUIDL",
    "USD1", "USDG", "AUSD", "HONEY", "NECT", "USDN", "SUSD0", "USDZ", "USDL", "MUSD",
}
STABLES -= {"EURC", "EURS", "AGEUR", "EURA"}
EUR_STABLES = {"EURC", "EURS", "AGEUR", "EURA", "EURE", "EURT", "CEUR", "EURCV", "EUROC", "EURO3", "EURI"}
ETH_FAMILY = {
    "ETH", "WETH", "STETH", "WSTETH", "RETH", "CBETH", "WEETH", "EETH", "EZETH", "RSETH",
    "METH", "SFRXETH", "FRXETH", "OSETH", "ETHX", "SWETH", "PUFETH", "ANKRETH", "LSETH",
    "RSWETH", "WOETH", "OETH", "APXETH", "PXETH", "UNIETH", "STONE", "BEDROCK",
}
BTC_FAMILY = {
    "BTC", "WBTC", "TBTC", "CBBTC", "LBTC", "SOLVBTC", "BTCB", "FBTC", "EBTC", "PUMPBTC",
    "UNIBTC", "SBTC", "WBTC.E", "BTC.B", "XBTC", "MBTC",
}
MAJORS = {
    "SOL", "WSOL", "JITOSOL", "MSOL", "BSOL", "JUPSOL", "BNB", "WBNB", "AVAX", "WAVAX",
    "MATIC", "POL", "WMATIC", "ARB", "OP", "LINK", "UNI", "AAVE", "LDO", "CRV", "CVX",
    "MKR", "SKY", "SUI", "APT", "TON", "DOT", "ATOM", "NEAR", "TRX", "XRP", "ADA", "DOGE",
    "LTC", "BCH", "FIL", "INJ", "TIA", "SEI", "JUP", "RAY", "ORCA", "AERO", "VELO", "CAKE",
    "GMX", "PENDLE", "ENA", "ETHFI", "EIGEN", "ONDO", "HYPE", "WHYPE", "S", "WS", "BERA",
    "WBERA", "MNT", "WMNT", "ZK", "STRK", "TAO", "RENDER", "FET", "WLD", "MORPHO", "COMP",
}

COINGECKO_IDS = {
    "ETH": "ethereum", "WETH": "ethereum", "BTC": "bitcoin", "WBTC": "wrapped-bitcoin",
    "SOL": "solana", "WSOL": "solana", "BNB": "binancecoin", "WBNB": "binancecoin",
    "AVAX": "avalanche-2", "WAVAX": "avalanche-2", "MATIC": "matic-network", "POL": "matic-network",
    "ARB": "arbitrum", "OP": "optimism", "LINK": "chainlink", "UNI": "uniswap", "AAVE": "aave",
    "USDC": "usd-coin", "USDT": "tether", "DAI": "dai", "SUI": "sui", "APT": "aptos",
}

# Annualised volatility of ln(Pa/Pb) by pair class when no history is available.
HEURISTIC_PAIR_SIGMA = {
    ("stable", "stable"): 0.02,
    ("eurstable", "eurstable"): 0.02,
    ("eurstable", "stable"): 0.08,      # EUR/USD FX volatility
    ("eth", "eurstable"): 0.72,
    ("btc", "eurstable"): 0.57,
    ("major", "eurstable"): 0.96,
    ("eth", "eth"): 0.06,
    ("btc", "btc"): 0.06,
    ("eth", "btc"): 0.40,
    ("eth", "stable"): 0.70,
    ("btc", "stable"): 0.55,
    ("major", "stable"): 0.95,
    ("major", "eth"): 0.80,
    ("major", "btc"): 0.85,
    ("major", "major"): 1.00,
}
EXOTIC_SIGMA = 1.40
SINGLE_ASSET_SIGMA = 0.0


def classify(symbol: str) -> str:
    s = symbol.upper().strip()
    if s in EUR_STABLES:
        return "eurstable"
    if s in STABLES or (s.startswith(("USD", "SUSD")) and len(s) <= 6):
        return "stable"
    if s in ETH_FAMILY:
        return "eth"
    if s in BTC_FAMILY:
        return "btc"
    if s in MAJORS:
        return "major"
    return "exotic"


def heuristic_sigma(tokens: Sequence[str]) -> Tuple[float, str]:
    """Return (sigma, label) for a pool from its token symbols alone."""
    toks = [t for t in tokens if t]
    if len(toks) <= 1:
        return SINGLE_ASSET_SIGMA, "heuristic:single"
    classes = [classify(t) for t in toks]
    if "exotic" in classes:
        return EXOTIC_SIGMA, "heuristic:exotic"
    worst = 0.0
    worst_label = ""
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            key = tuple(sorted((classes[i], classes[j])))
            key = _canon(key)
            sig = HEURISTIC_PAIR_SIGMA.get(key, EXOTIC_SIGMA)
            if sig >= worst:
                worst, worst_label = sig, f"{key[0]}/{key[1]}"
    return worst, f"heuristic:{worst_label}"


def _canon(key: Tuple[str, str]) -> Tuple[str, str]:
    order = {"stable": 4, "eurstable": 3, "eth": 1, "btc": 2, "major": 0}
    a, b = sorted(key, key=lambda c: order.get(c, 9))
    return (a, b)


def log_returns(prices: Sequence[float]) -> List[float]:
    out = []
    for a, b in zip(prices, prices[1:]):
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def realised_vol(prices: Sequence[float], periods_per_year: float = 365.0) -> Optional[float]:
    r = log_returns(prices)
    if len(r) < 10:
        return None
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
    return math.sqrt(var * periods_per_year)


def correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    ra, rb = log_returns(a), log_returns(b)
    n = min(len(ra), len(rb))
    if n < 10:
        return None
    ra, rb = ra[-n:], rb[-n:]
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va <= 0 or vb <= 0:
        return None
    return cov / math.sqrt(va * vb)


def ratio_sigma_from_history(pa: Sequence[float], pb: Sequence[float]) -> Optional[float]:
    """Vol of ln(Pa/Pb) straight from the two aligned series."""
    n = min(len(pa), len(pb))
    if n < 11:
        return None
    ratio = [x / y for x, y in zip(pa[-n:], pb[-n:]) if y > 0]
    return realised_vol(ratio)


class PriceSource:
    def history(self, keys: Sequence[str], days: int) -> Dict[str, List[float]]:  # pragma: no cover
        raise NotImplementedError

    def current(self, keys: Sequence[str]) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError


class DefiLlamaPrices(PriceSource):
    """coins.llama.fi: keys look like 'ethereum:0x...' or 'coingecko:ethereum'."""

    def __init__(self, base_url: str = "https://coins.llama.fi", timeout_s: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.errors: List[str] = []

    def _get(self, path: str) -> Dict:
        req = urllib.request.Request(self.base_url + path, headers={"User-Agent": "lp_bot/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(300).decode("utf-8", "replace").strip()
            except Exception:
                pass
            raise RuntimeError(f"HTTP {exc.code} for {path[:120]} :: {body}") from None

    @staticmethod
    def _series(payload: Dict) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for key, rec in (payload.get("coins") or {}).items():
            pts = sorted(rec.get("prices") or [], key=lambda p: p.get("timestamp", 0))
            out[key] = [float(p["price"]) for p in pts if p.get("price")]
        return out

    def _chart(self, keys: List[str], days: int) -> Dict[str, List[float]]:
        """/chart/{coins}: one request for up to 10 coins, daily points ending yesterday."""
        start = int(time.time()) - (days + 1) * 86400
        q = urllib.parse.urlencode({"start": start, "span": days, "period": "1d"})
        chunk = urllib.parse.quote(",".join(keys), safe=",:")
        return self._series(self._get(f"/chart/{chunk}?{q}"))

    def _batch_historical(self, keys: List[str], days: int) -> Dict[str, List[float]]:
        """/batchHistorical: fallback that asks for explicit daily timestamps."""
        now = int(time.time()) - 3600
        stamps = [now - i * 86400 for i in range(days, 0, -1)]
        coins = json.dumps({k: stamps for k in keys}, separators=(",", ":"))
        q = urllib.parse.urlencode({"coins": coins, "searchWidth": "6h"})
        return self._series(self._get(f"/batchHistorical?{q}"))

    def history(self, keys: Sequence[str], days: int) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        keys = list(dict.fromkeys(k for k in keys if k))
        use_batch = False
        for i in range(0, len(keys), 10):
            chunk = keys[i:i + 10]
            if not use_batch:
                try:
                    out.update(self._chart(chunk, days))
                    continue
                except Exception as exc:  # switch to the fallback endpoint for the rest
                    self.errors.append(f"chart endpoint failed, falling back to batchHistorical: {exc}")
                    use_batch = True
            for j in range(0, len(chunk), 3):
                sub = chunk[j:j + 3]
                try:
                    out.update(self._batch_historical(sub, days))
                except Exception as exc:  # one bad batch must not blank out every other pool
                    self.errors.append(f"batchHistorical {sub[0][:40]}...: {exc}")
        return out

    def current(self, keys: Sequence[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        keys = list(dict.fromkeys(keys))
        for i in range(0, len(keys), 50):
            chunk = ",".join(keys[i:i + 50])
            payload = self._get(f"/prices/current/{urllib.parse.quote(chunk, safe=',:')}")
            for key, rec in (payload.get("coins") or {}).items():
                if rec.get("price") is not None:
                    out[key] = float(rec["price"])
        return out


class FixturePrices(PriceSource):
    """Prices from a dict {key: [p0, p1, ...]} (history) - last element is 'current'."""

    def __init__(self, series: Dict[str, List[float]]):
        self.series = {k: list(v) for k, v in series.items()}

    def history(self, keys: Sequence[str], days: int) -> Dict[str, List[float]]:
        return {k: self.series[k][-days:] for k in keys if k in self.series}

    def current(self, keys: Sequence[str]) -> Dict[str, float]:
        return {k: self.series[k][-1] for k in keys if k in self.series and self.series[k]}


class NullPrices(PriceSource):
    def history(self, keys, days):
        return {}

    def current(self, keys):
        return {}


def price_keys(pool: Pool) -> List[str]:
    """DefiLlama coin keys for the pool's legs. Uses on-chain addresses when the source
    supplied them, else a CoinGecko id for well-known symbols, else nothing."""
    chain = pool.chain.lower().replace(" ", "-")
    keys: List[str] = []
    toks = pool.tokens
    addrs = pool.underlying_tokens
    if addrs and len(addrs) == len(toks):
        return [f"{chain}:{a}" for a in addrs]
    for t in toks:
        cg = COINGECKO_IDS.get(t.upper())
        keys.append(f"coingecko:{cg}" if cg else "")
    return keys


class VolEstimator:
    """Pair-volatility with history first and heuristics as fallback. Caches per pool."""

    def __init__(self, prices: PriceSource, history_days: int = 90):
        self.prices = prices
        self.history_days = history_days
        self._hist_cache: Dict[str, List[float]] = {}
        self._pool_cache: Dict[str, Tuple[float, str, float]] = {}
        self.warnings: List[str] = []

    def stats(self) -> Dict[str, int]:
        """How many scored pools used realised history versus a heuristic."""
        hist = sum(1 for v in self._pool_cache.values() if v[1] == "history")
        return {"history": hist, "heuristic": len(self._pool_cache) - hist}

    def prefetch(self, pools: Iterable[Pool]) -> None:
        wanted = set()
        for p in pools:
            for k in price_keys(p):
                if k and k not in self._hist_cache:
                    wanted.add(k)
        if not wanted:
            return
        try:
            fetched = self.prices.history(sorted(wanted), self.history_days)
        except Exception as exc:  # network failure -> heuristics only
            fetched = {}
            self.warnings.append(f"price history unavailable: {exc}")
        errs = getattr(self.prices, "errors", None)
        if errs:
            self.warnings.extend(errs)
            del errs[:]
        for k in wanted:
            self._hist_cache[k] = fetched.get(k, [])

    def legs(self, pool: Pool) -> Tuple[Optional[str], Optional[str]]:
        """The two legs whose ratio drives IL: the most volatile pair in the pool."""
        toks = pool.tokens
        keys = price_keys(pool)
        if len(toks) < 2:
            return None, None
        best, best_pair = -1.0, (None, None)
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                sig, _ = heuristic_sigma([toks[i], toks[j]])
                if sig > best:
                    best = sig
                    ki = keys[i] if i < len(keys) else None
                    kj = keys[j] if j < len(keys) else None
                    best_pair = (ki or None, kj or None)
        return best_pair

    def estimate(self, pool: Pool) -> Tuple[float, str, float]:
        """Return (sigma, source, current_price_ratio)."""
        if pool.pool_id in self._pool_cache:
            return self._pool_cache[pool.pool_id]
        toks = pool.tokens
        h_sigma, h_label = heuristic_sigma(toks)
        result = (h_sigma, h_label, 1.0)
        if len(toks) >= 2:
            ka, kb = self.legs(pool)
            if ka and kb:
                if ka not in self._hist_cache or kb not in self._hist_cache:
                    self.prefetch([pool])
                pa, pb = self._hist_cache.get(ka, []), self._hist_cache.get(kb, [])
                sig = ratio_sigma_from_history(pa, pb)
                if sig is not None and pa and pb:
                    ratio = pa[-1] / pb[-1] if pb[-1] > 0 else 1.0
                    result = (sig, "history", ratio)
        self._pool_cache[pool.pool_id] = result
        return result

    def current_ratio(self, pool: Pool) -> Optional[float]:
        """Fresh price ratio for monitoring an open position (None if unavailable)."""
        ka, kb = self.legs(pool)
        if not ka or not kb:
            return None
        try:
            cur = self.prices.current([ka, kb])
        except Exception:
            return None
        if ka in cur and kb in cur and cur[kb] > 0:
            return cur[ka] / cur[kb]
        return None
