"""Token prices and pair-volatility estimation.

Volatility of the price *ratio* of a pool's two legs is the single input that matters most
for impermanent loss. We try, in order:

1. Realised volatility from daily price history (DefiLlama coins API, any chain), aligned by
   calendar day so a missing day or a different endpoint cannot shift one leg.
2. A heuristic by token class (stable / ETH-family / BTC-family / SOL-family / major / exotic).

Every estimate carries its provenance so reports can say which one was used. Histories are
refreshed every ``refresh_s`` seconds and coins that returned nothing are retried after
``retry_s`` seconds, so a long-running loop never freezes on stale or failed data.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Pool

# ----------------------------------------------------------------------------- classes
STABLES = {
    "USDC", "USDT", "DAI", "USDE", "SUSDE", "FRAX", "LUSD", "GHO", "CRVUSD", "USDS", "SUSDS",
    "PYUSD", "SUSD", "TUSD", "USDD", "MIM", "FDUSD", "USD0", "USDC.E", "USDT.E", "USDBC",
    "USDB", "DOLA", "BOLD", "GUSD", "USDP", "USDM", "USDY", "USDA", "USDX", "USR", "RLUSD",
    "DEUSD", "SFRAX", "SDAI", "USD+", "USDT0", "USD₮0", "USD₮", "BUIDL", "USD1", "USDG", "AUSD",
    "HONEY", "NECT", "USDN", "SUSD0", "USDZ", "USDL", "MUSD", "MSUSD", "FRXUSD", "SFRXUSD",
    "USDAI", "YUSD", "SYRUPUSDC", "AVUSD", "SAVUSD", "WUSDM", "USDF", "EUSD",
}
# Stablecoins that accrue yield in their price (worth more than $1 over time by design).
YIELD_BEARING_STABLES = {
    "SUSDE", "SDAI", "SUSDS", "SFRAX", "SFRXUSD", "USDY", "SYRUPUSDC", "SAVUSD", "WUSDM", "YUSD",
    "SUSD0", "USD0++", "STUSD", "SUSDA", "SDOLA", "SCRVUSD", "WSTUSR", "SUSDX", "SUSDF", "SBOLD",
    "STKGHO", "SGHO", "SYRUPUSDT", "SUSDAI", "SNUSD",
}
STABLES |= YIELD_BEARING_STABLES
EUR_STABLES = {"EURC", "EURS", "AGEUR", "EURA", "EURE", "EURT", "CEUR", "EURCV", "EUROC", "EURO3", "EURI"}
ETH_FAMILY = {
    "ETH", "WETH", "STETH", "WSTETH", "RETH", "CBETH", "WEETH", "EETH", "EZETH", "RSETH",
    "METH", "SFRXETH", "FRXETH", "OSETH", "ETHX", "SWETH", "PUFETH", "ANKRETH", "LSETH",
    "RSWETH", "WOETH", "OETH", "APXETH", "PXETH", "UNIETH", "STONE", "BEDROCK", "MSETH",
    "WRSETH", "SUPERETH",
}
BTC_FAMILY = {
    "BTC", "WBTC", "TBTC", "CBBTC", "LBTC", "SOLVBTC", "BTCB", "FBTC", "EBTC", "PUMPBTC",
    "UNIBTC", "SBTC", "BTC.B", "XBTC", "MBTC", "UBTC", "KBTC", "ZBTC",
}
SOL_FAMILY = {
    "SOL", "WSOL", "JITOSOL", "MSOL", "BSOL", "JUPSOL", "INF", "BBSOL", "HSOL", "VSOL",
    "DSOL", "BNSOL", "JSOL", "LAINESOL", "STSOL", "PSOL", "HUBSOL", "STRONGSOL",
}
MAJORS = {
    "BNB", "WBNB", "AVAX", "WAVAX", "SAVAX", "MATIC", "POL", "WMATIC", "WPOL", "ARB", "OP",
    "LINK", "UNI", "AAVE", "LDO", "CRV", "CVX", "MKR", "SKY", "SUI", "APT", "TON", "DOT",
    "ATOM", "NEAR", "TRX", "XRP", "FXRP", "ADA", "DOGE", "LTC", "BCH", "FIL", "INJ", "TIA",
    "SEI", "WSEI", "JUP", "RAY", "ORCA", "AERO", "VELO", "CAKE", "GMX", "PENDLE", "ENA",
    "ETHFI", "EIGEN", "ONDO", "HYPE", "WHYPE", "KHYPE", "S", "WS", "BERA", "WBERA", "MNT",
    "WMNT", "ZK", "STRK", "TAO", "RENDER", "FET", "WLD", "MORPHO", "COMP", "FTM", "WFTM",
}
FX_CODES = {
    "CAD", "JPY", "CHF", "GBP", "EUR", "AUD", "NZD", "MXN", "BRL", "SGD", "HKD", "KRW", "TRY",
    "ZAR", "CNY", "CNH", "INR", "SEK", "NOK", "PLN", "IDR", "PHP", "THB", "XAU", "XAG",
}
WRAPPED_TO_BASE = {
    "WETH": "ETH", "WSOL": "SOL", "WBTC": "BTC", "WAVAX": "AVAX", "WBNB": "BNB", "WMATIC": "MATIC",
    "WPOL": "POL", "WHYPE": "HYPE", "WS": "S", "WMNT": "MNT", "WBERA": "BERA", "WSEI": "SEI",
    "WFTM": "FTM", "FXRP": "XRP", "BTC.B": "BTC", "BTCB": "BTC", "USDC.E": "USDC", "USDT.E": "USDT",
    "USDBC": "USDC", "USDT0": "USDT", "USD₮0": "USDT", "USD₮": "USDT",
}

COINGECKO_IDS = {
    "ETH": "ethereum", "WETH": "ethereum", "BTC": "bitcoin", "WBTC": "wrapped-bitcoin",
    "SOL": "solana", "WSOL": "solana", "BNB": "binancecoin", "WBNB": "binancecoin",
    "AVAX": "avalanche-2", "WAVAX": "avalanche-2", "MATIC": "matic-network", "WMATIC": "matic-network",
    "POL": "matic-network", "ARB": "arbitrum", "OP": "optimism", "LINK": "chainlink",
    "UNI": "uniswap", "AAVE": "aave", "USDC": "usd-coin", "USDT": "tether", "DAI": "dai",
    "SUI": "sui", "APT": "aptos", "USDE": "ethena-usde", "STETH": "staked-ether",
    "WSTETH": "wrapped-steth", "CBBTC": "coinbase-wrapped-btc", "EURC": "euro-coin",
    "AERO": "aerodrome-finance", "GRAIL": "camelot-token", "BRETT": "based-brett",
}

# yields-API chain display labels -> coins-API chain keys (they differ for several chains)
LLAMA_CHAIN_KEYS = {
    "op mainnet": "optimism", "optimism": "optimism", "avalanche": "avax", "gnosis": "xdai",
    "zksync era": "era", "polygon zkevm": "polygon_zkevm", "hyperliquid l1": "hyperliquid",
    "plume mainnet": "plume_mainnet", "xrpl evm": "xrplevm", "etherlink": "etlk", "opbnb": "op_bnb",
    "cronos zkevm": "cronos_zkevm", "robinhood chain": "robinhood", "zklink nova": "zklink",
    "arbitrum nova": "arbitrum_nova", "binance": "bsc", "bsc": "bsc", "world chain": "wc",
    "rootstock": "rsk", "immutable zkevm": "imx", "x layer": "xlayer", "kaia": "klaytn",
    "bitlayer": "btr", "polygon": "polygon", "sei": "sei", "sonic": "sonic",
}

# Annualised volatility of ln(Pa/Pb) by pair class when no history is available.
HEURISTIC_PAIR_SIGMA = {
    ("stable", "stable"): 0.02,
    ("eurstable", "eurstable"): 0.02,
    ("eurstable", "stable"): 0.08,      # EUR/USD FX volatility
    ("fx", "stable"): 0.10,
    ("eth", "eurstable"): 0.72,
    ("btc", "eurstable"): 0.57,
    ("sol", "eurstable"): 0.82,
    ("major", "eurstable"): 0.96,
    ("eth", "eth"): 0.06,
    ("btc", "btc"): 0.06,
    ("sol", "sol"): 0.06,
    ("eth", "btc"): 0.40,
    ("sol", "eth"): 0.70,
    ("sol", "btc"): 0.75,
    ("eth", "stable"): 0.70,
    ("btc", "stable"): 0.55,
    ("sol", "stable"): 0.80,
    ("major", "stable"): 0.95,
    ("major", "eth"): 0.80,
    ("major", "btc"): 0.85,
    ("major", "sol"): 0.90,
    ("major", "major"): 1.00,
}
EXOTIC_SIGMA = 1.40
SINGLE_ASSET_SIGMA = 0.0


def _lookup_class(s: str) -> Optional[str]:
    if s in EUR_STABLES:
        return "eurstable"
    if len(s) == 6 and s[:3] in FX_CODES | {"USD"} and s[3:] in FX_CODES | {"USD"}:
        return "fx"
    if s in STABLES:
        return "stable"
    if s in ETH_FAMILY:
        return "eth"
    if s in BTC_FAMILY:
        return "btc"
    if s in SOL_FAMILY:
        return "sol"
    if s in MAJORS:
        return "major"
    return None


def classify(symbol: str) -> str:
    """stable | eurstable | fx | eth | btc | sol | major | exotic. Bridge suffixes such as
    ``.E`` and ``.B`` are stripped when the full symbol is unknown."""
    s = symbol.upper().strip()
    for cand in (s, s.split(".")[0]):
        c = _lookup_class(cand)
        if c:
            return c
    base = s.split(".")[0]
    if base.startswith(("USD", "SUSD")) and len(base) <= 6:
        return "stable"
    return "exotic"


def normalise_symbol(symbol: str) -> str:
    """Canonical asset name used to compare pairs across pools (WETH and ETH are the same)."""
    s = symbol.upper().strip()
    if s in WRAPPED_TO_BASE:
        return WRAPPED_TO_BASE[s]
    base = s.split(".")[0]
    return WRAPPED_TO_BASE.get(base, base)


def pair_key(tokens: Sequence[str]) -> str:
    return "/".join(sorted({normalise_symbol(t) for t in tokens if t}))


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
            key = _canon((classes[i], classes[j]))
            sig = HEURISTIC_PAIR_SIGMA.get(key, EXOTIC_SIGMA)
            if sig >= worst:
                worst, worst_label = sig, f"{key[0]}/{key[1]}"
    return worst, f"heuristic:{worst_label}"


def _canon(key: Tuple[str, str]) -> Tuple[str, str]:
    order = {"major": 0, "eth": 1, "btc": 2, "sol": 3, "fx": 4, "eurstable": 5, "stable": 6}
    a, b = sorted(key, key=lambda c: order.get(c, 9))
    if (a, b) not in HEURISTIC_PAIR_SIGMA and (b, a) in HEURISTIC_PAIR_SIGMA:
        return (b, a)
    return (a, b)


# ----------------------------------------------------------------------------- statistics
Series = Sequence  # list of prices, or list of (day, price) tuples


def _as_daymap(series: Series) -> Dict[int, float]:
    """Map day -> price. Plain price lists are aligned at their last element."""
    if not series:
        return {}
    first = series[0]
    if isinstance(first, (tuple, list)):
        return {int(d): float(p) for d, p in series if p and p > 0}
    n = len(series)
    return {i - n: float(p) for i, p in enumerate(series) if p and p > 0}


def _vol_from_daymap(dm: Dict[int, float], periods_per_year: float = 365.0) -> Optional[float]:
    days = sorted(dm)
    xs = []
    for d1, d2 in zip(days, days[1:]):
        gap = d2 - d1
        if gap <= 0:
            continue
        xs.append(math.log(dm[d2] / dm[d1]) / math.sqrt(gap))
    if len(xs) < 10:
        return None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var * periods_per_year)


def log_returns(prices: Sequence[float]) -> List[float]:
    out = []
    for a, b in zip(prices, prices[1:]):
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def realised_vol(prices: Series, periods_per_year: float = 365.0) -> Optional[float]:
    return _vol_from_daymap(_as_daymap(prices), periods_per_year)


def ratio_sigma_from_history(pa: Series, pb: Series) -> Optional[float]:
    """Vol of ln(Pa/Pb) using only the days both legs have a price."""
    a, b = _as_daymap(pa), _as_daymap(pb)
    common = sorted(set(a) & set(b))
    if len(common) < 11:
        return None
    return _vol_from_daymap({d: a[d] / b[d] for d in common})


def _remap_keys(requested: Iterable[str], got: Dict) -> Dict:
    """The coins API may change the case of an address; map results back to every key we
    asked for that matches, case-insensitively."""
    low: Dict[str, List[str]] = {}
    for k in requested:
        low.setdefault(k.lower(), []).append(k)
    out: Dict = {}
    for k, v in got.items():
        for orig in low.get(k.lower(), [k]):
            out[orig] = v
    return out


def ratio_drift_from_history(pa: Series, pb: Series) -> float:
    """Annualised drift of ln(Pa/Pb) over the common days (0 if too little data)."""
    a, b = _as_daymap(pa), _as_daymap(pb)
    common = sorted(set(a) & set(b))
    if len(common) < 11 or common[-1] == common[0]:
        return 0.0
    r0 = a[common[0]] / b[common[0]]
    r1 = a[common[-1]] / b[common[-1]]
    return math.log(r1 / r0) / (common[-1] - common[0]) * 365.0


# ----------------------------------------------------------------------------- sources
class HttpStatusError(RuntimeError):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code

    @property
    def is_client_error(self) -> bool:  # bad request for this key: not an outage
        return 400 <= self.code < 500 and self.code not in (401, 403, 429)


class PriceSource:
    errors: List[str]
    attempted: Optional[set] = None   # keys actually requested by the last history() call

    def historical(self, keys: Sequence[str], ts: int) -> Dict[str, float]:
        return {}

    def history(self, keys: Sequence[str], days: int, deadline: Optional[float] = None) -> Dict[str, list]:  # pragma: no cover
        raise NotImplementedError

    def current(self, keys: Sequence[str]) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError


class DefiLlamaPrices(PriceSource):
    """coins.llama.fi: keys look like 'ethereum:0x...' or 'coingecko:ethereum'."""

    def __init__(self, base_url: str = "https://coins.llama.fi", timeout_s: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.errors: List[str] = []
        self.bad_keys: List[str] = []

    def _get(self, path: str) -> Dict:
        req = urllib.request.Request(self.base_url + path, headers={"User-Agent": "lp_bot/0.2"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(300).decode("utf-8", "replace").strip()
            except Exception:
                pass
            raise HttpStatusError(exc.code, f"HTTP {exc.code} for {path[:120]} :: {body}") from None

    @staticmethod
    def _series(payload: Dict, requested: Sequence[str]) -> Dict[str, List[Tuple[int, float]]]:
        out: Dict[str, List[Tuple[int, float]]] = {}
        for key, rec in (payload.get("coins") or {}).items():
            pts = []
            for p in rec.get("prices") or []:
                ts, px = p.get("timestamp"), p.get("price")
                if ts is None or not px or px <= 0:
                    continue
                pts.append((int(round(float(ts) / 86400.0)), float(px)))
            pts.sort()
            out[key] = pts
        return _remap_keys(requested, out)

    @staticmethod
    def _grid(days: int) -> Tuple[int, List[int]]:
        """One daily grid (UTC midnights) shared by both endpoints."""
        today = int(time.time() // 86400) * 86400
        start = today - days * 86400
        return start, [start + i * 86400 for i in range(days)]

    def _chart(self, keys: List[str], days: int) -> Dict[str, list]:
        start, _ = self._grid(days)
        q = urllib.parse.urlencode({"start": start, "span": days, "period": "1d", "searchWidth": "12h"})
        chunk = urllib.parse.quote(",".join(keys), safe=",:")
        return self._series(self._get(f"/chart/{chunk}?{q}"), keys)

    def _batch_historical(self, keys: List[str], days: int) -> Dict[str, list]:
        _, stamps = self._grid(days)
        coins = json.dumps({k: stamps for k in keys}, separators=(",", ":"))
        q = urllib.parse.urlencode({"coins": coins, "searchWidth": "12h"})
        return self._series(self._get(f"/batchHistorical?{q}"), keys)

    def history(self, keys: Sequence[str], days: int, deadline: Optional[float] = None) -> Dict[str, list]:
        """Chunked /chart first. Keys missing from a chunk (failed request or absent from a
        200 response) are retried one at a time, then via /batchHistorical. Stops early at the
        ``deadline`` (time.monotonic) or after repeated failures, returning what it has."""
        out: Dict[str, list] = {}
        keys = list(dict.fromkeys(k for k in keys if k))
        self.bad_keys = []
        self.attempted = set()
        consecutive_failures = 0  # transport failures only; a rejected key is not an outage
        client_streak = 0         # ...unless every request is rejected and none has succeeded
        any_success = False

        def out_of_time() -> bool:
            return (deadline is not None and time.monotonic() > deadline) or consecutive_failures >= 6

        def failed(exc: Exception) -> None:
            nonlocal consecutive_failures, client_streak
            if isinstance(exc, HttpStatusError) and exc.is_client_error:
                client_streak += 1
                if not any_success and client_streak >= 10:
                    consecutive_failures = 6  # blanket rejection: treat as an outage
            else:
                consecutive_failures += 1

        def succeeded() -> None:
            nonlocal consecutive_failures, client_streak, any_success
            consecutive_failures, client_streak, any_success = 0, 0, True

        for i in range(0, len(keys), 10):
            if out_of_time():
                self.errors.append(f"history fetch stopped early; {len(keys) - i} coins deferred")
                break
            chunk = keys[i:i + 10]
            try:
                got = self._chart(chunk, days)
                out.update({k: v for k, v in got.items() if v})
                succeeded()
            except Exception as exc:
                failed(exc)
                if not isinstance(exc, HttpStatusError) or not exc.is_client_error:
                    self.errors.append(f"chart chunk failed: {exc}")
            for key in chunk:
                if key in out:
                    self.attempted.add(key)
                    continue
                if out_of_time():
                    continue
                self.attempted.add(key)
                ok = False
                for fetch in (self._chart, self._batch_historical):
                    try:
                        got = fetch([key], days)
                        succeeded()
                        if got.get(key):
                            out[key] = got[key]
                            ok = True
                            break
                    except Exception as exc:
                        failed(exc)
                        if len(self.errors) < 20:
                            self.errors.append(f"{key}: {exc}")
                if not ok:
                    self.bad_keys.append(key)
        if self.bad_keys:
            self.errors.append(f"{len(self.bad_keys)} coin keys returned no history")
        return out

    def _prices_at(self, path_prefix: str, chunk: List[str], suffix: str = "") -> Dict[str, float]:
        payload = self._get(f"{path_prefix}/{urllib.parse.quote(','.join(chunk), safe=',:')}{suffix}")
        got = {k: float(r["price"]) for k, r in (payload.get("coins") or {}).items()
               if r.get("price") is not None and float(r["price"]) > 0}
        return _remap_keys(chunk, got)

    def _batched(self, path_prefix: str, keys: Sequence[str], suffix: str = "") -> Dict[str, float]:
        """Fetch in chunks; a chunk that fails is retried key by key so one rejected coin
        cannot blank out the others."""
        out: Dict[str, float] = {}
        keys = list(dict.fromkeys(k for k in keys if k))
        for i in range(0, len(keys), 50):
            chunk = keys[i:i + 50]
            try:
                out.update(self._prices_at(path_prefix, chunk, suffix))
                continue
            except Exception as exc:
                if len(chunk) == 1:
                    self.errors.append(f"price for {chunk[0]} failed: {exc}")
                    continue
                transport = not (isinstance(exc, HttpStatusError) and exc.is_client_error)
                if transport:
                    self.errors.append(f"price request failed: {exc}")
                    continue
            for key in chunk:
                try:
                    out.update(self._prices_at(path_prefix, [key], suffix))
                except Exception as exc:
                    self.errors.append(f"price for {key} failed: {exc}")
        return out

    def current(self, keys: Sequence[str]) -> Dict[str, float]:
        return self._batched("/prices/current", keys)

    def historical(self, keys: Sequence[str], ts: int) -> Dict[str, float]:
        """USD prices at a past timestamp (used to give pre-upgrade positions a real basis)."""
        return self._batched(f"/prices/historical/{int(ts)}", keys, "?searchWidth=12h")


class FixturePrices(PriceSource):
    """Prices from a dict {key: [p0, p1, ...]} (history); the last element is 'current'.
    Tests move prices by appending to or editing ``series``."""

    def __init__(self, series: Dict[str, List[float]]):
        self.series = {k: list(v) for k, v in series.items()}
        self.errors: List[str] = []
        self.fail_current = False
        self.fail_history = False
        self.at: Dict[int, Dict[str, float]] = {}   # optional {timestamp: {key: price}}
        self.reject: set = set()                    # keys the 'API' refuses

    def historical(self, keys, ts):
        best = min(self.at, key=lambda t: abs(t - ts)) if self.at else None
        if best is None or abs(best - ts) > 12 * 3600:
            return {}
        return {k: v for k, v in self.at[best].items() if k in keys}

    def history(self, keys, days, deadline=None):
        self.attempted = set(keys)
        if self.fail_history:
            return {}
        return {k: self.series[k][-days:] for k in keys if k in self.series and k not in self.reject}

    def current(self, keys):
        if self.fail_current:
            return {}
        return {k: self.series[k][-1] for k in keys if k in self.series and self.series[k] and k not in self.reject}


class NullPrices(PriceSource):
    def __init__(self):
        self.errors: List[str] = []

    def history(self, keys, days, deadline=None):
        return {}

    def current(self, keys):
        return {}


def chain_key(chain: str) -> str:
    c = chain.strip().lower()
    return LLAMA_CHAIN_KEYS.get(c, c.replace(" ", "_"))


def price_keys(pool: Pool) -> List[str]:
    """coins-API keys for the pool's legs: on-chain addresses when the source supplied one per
    token, else a CoinGecko id for well-known symbols, else ''."""
    chain = chain_key(pool.chain)
    toks = pool.tokens
    addrs = pool.underlying_tokens
    if addrs and len(addrs) == len(toks):
        return [f"{chain}:{a}" for a in addrs]
    keys: List[str] = []
    for t in toks:
        cg = COINGECKO_IDS.get(t.upper()) or COINGECKO_IDS.get(normalise_symbol(t))
        keys.append(f"coingecko:{cg}" if cg else "")
    return keys


def symbol_keys(pool: Pool) -> List[str]:
    """CoinGecko-id keys from symbols alone (fallback when address keys fail)."""
    out = []
    for t in pool.tokens:
        cg = COINGECKO_IDS.get(t.upper()) or COINGECKO_IDS.get(normalise_symbol(t))
        out.append(f"coingecko:{cg}" if cg else "")
    return out


def peg_kind(symbol: str) -> Optional[str]:
    """'yield' for yield-bearing USD stables, 'usd' for plain USD stables, else None."""
    s = symbol.upper().strip()
    if s in YIELD_BEARING_STABLES or s.split(".")[0] in YIELD_BEARING_STABLES:
        return "yield"
    return "usd" if classify(symbol) == "stable" else None


class VolEstimator:
    """Pair volatility with history first and heuristics as fallback.

    History is cached per coin key with its fetch time: refreshed after ``refresh_s``, and a
    coin that returned nothing is retried after ``retry_s``. Estimates themselves are cheap
    and recomputed on every call, so new history is picked up immediately."""

    def __init__(self, prices: PriceSource, history_days: int = 90, refresh_s: float = 86400.0,
                 retry_s: float = 6 * 3600.0, clock: Callable[[], float] = time.time):
        self.prices = prices
        self.history_days = history_days
        self.refresh_s = refresh_s
        self.retry_s = retry_s
        self.clock = clock
        self._hist: Dict[str, Tuple[float, list]] = {}
        self._sources: Dict[str, str] = {}
        self.warnings: List[str] = []

    # ---- cache -----------------------------------------------------------------------
    def _needs_fetch(self, key: str, now: float) -> bool:
        if key not in self._hist:
            return True
        fetched_at, series = self._hist[key]
        age = now - fetched_at
        return age >= (self.refresh_s if series else self.retry_s)

    def _drain_source_errors(self) -> None:
        errs = getattr(self.prices, "errors", None)
        if errs:
            self.warnings.extend(errs)
            del errs[:]

    def prefetch(self, pools: Iterable[Pool], budget_s: Optional[float] = None) -> None:
        now = self.clock()
        wanted = set()
        for p in pools:
            ka, kb, _, _ = self.legs(p)
            for k in (ka, kb):
                if k and self._needs_fetch(k, now):
                    wanted.add(k)
        if not wanted:
            return
        deadline = time.monotonic() + budget_s if budget_s else None
        try:
            fetched = self.prices.history(sorted(wanted), self.history_days, deadline=deadline)
        except Exception as exc:  # network failure -> heuristics this round, retry later
            fetched = {}
            self.warnings.append(f"price history unavailable: {exc}")
        self._drain_source_errors()
        attempted = getattr(self.prices, "attempted", None)
        for k in wanted:
            if attempted is not None and k not in attempted and not fetched.get(k):
                continue  # deferred by the time budget: try again next cycle, not in retry_s
            series = fetched.get(k) or []
            if series:
                self._hist[k] = (now, list(series))
            elif k in self._hist and self._hist[k][1]:
                # failed refresh: keep the last good series, retry after retry_s
                self._hist[k] = (now - self.refresh_s + self.retry_s, self._hist[k][1])
            else:
                self._hist[k] = (now, [])

    def reset_stats(self) -> None:
        self._sources = {}

    def stats(self) -> Dict[str, int]:
        hist = sum(1 for v in self._sources.values() if v == "history")
        return {"history": hist, "heuristic": len(self._sources) - hist}

    # ---- legs and estimates ------------------------------------------------------------
    def legs(self, pool: Pool) -> Tuple[Optional[str], Optional[str], str, str]:
        """(key_a, key_b, symbol_a, symbol_b) for the most volatile pair in the pool.
        Deterministic, so a position always sees the same orientation."""
        toks = pool.tokens
        if len(toks) < 2:
            return None, None, toks[0] if toks else "", ""
        keys = price_keys(pool)
        best, pick = -1.0, (0, 1)
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                sig, _ = heuristic_sigma([toks[i], toks[j]])
                if sig > best:
                    best, pick = sig, (i, j)
        i, j = pick
        ka = keys[i] if i < len(keys) and keys[i] else None
        kb = keys[j] if j < len(keys) and keys[j] else None
        return ka, kb, toks[i], toks[j]

    def leg_candidates(self, pool: Pool) -> List[Tuple[str, str, str, str]]:
        """Address-based legs first, then CoinGecko-symbol legs if they differ."""
        ka, kb, sa, sb = self.legs(pool)
        out = [(ka, kb, sa, sb)] if ka and kb else []
        toks = pool.tokens
        if len(toks) >= 2 and sa and sb:
            sk = symbol_keys(pool)
            ia, ib = toks.index(sa), toks.index(sb, toks.index(sa) + 1) if sa == sb else toks.index(sb)
            fa = sk[ia] if ia < len(sk) else ""
            fb = sk[ib] if ib < len(sk) else ""
            # never price two different tokens (e.g. USDC.e and USDC) through one id
            if fa and fb and fa != fb and (fa, fb) != (ka, kb):
                out.append((fa, fb, sa, sb))
        return out

    def drift(self, pool: Pool) -> float:
        """Annualised drift of the pair's price ratio from history (0 when unknown)."""
        ka, kb, _, _ = self.legs(pool)
        if not ka or not kb:
            return 0.0
        return ratio_drift_from_history(self._hist.get(ka, (0, []))[1], self._hist.get(kb, (0, []))[1])

    def historical_prices(self, keys: Sequence[str], ts: int) -> Dict[str, float]:
        keys = [k for k in keys if k]
        if not keys:
            return {}
        try:
            got = self.prices.historical(keys, ts)
        except Exception as exc:
            self.warnings.append(f"historical prices unavailable: {exc}")
            got = {}
        self._drain_source_errors()
        return got

    def estimate(self, pool: Pool) -> Tuple[float, str]:
        """Return (sigma, source) where source is 'history' or 'heuristic:<label>'."""
        toks = pool.tokens
        sigma, source = heuristic_sigma(toks)
        if len(toks) >= 2:
            ka, kb, _, _ = self.legs(pool)
            if ka and kb:
                # no fetching here: callers prefetch under a time budget first
                pa = self._hist.get(ka, (0, []))[1]
                pb = self._hist.get(kb, (0, []))[1]
                sig = ratio_sigma_from_history(pa, pb)
                if sig is not None:
                    sigma, source = sig, "history"
        self._sources[pool.pool_id] = source
        return sigma, source

    # ---- current prices ----------------------------------------------------------------
    def current_prices(self, keys: Sequence[str]) -> Dict[str, float]:
        keys = [k for k in keys if k]
        if not keys:
            return {}
        try:
            got = self.prices.current(keys)
        except Exception as exc:
            self.warnings.append(f"current prices unavailable: {exc}")
            got = {}
        self._drain_source_errors()
        return got
