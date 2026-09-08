"""Plain data records shared by the scanner, scorer, manager, and executors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Pool:
    """One liquidity pool as reported by a yield source (DefiLlama shape)."""

    pool_id: str
    chain: str
    project: str
    symbol: str
    tvl_usd: float
    apy: float                        # total APY, percent
    apy_base: Optional[float] = None  # fee APY, percent
    apy_reward: Optional[float] = None  # emissions APY, percent
    apy_base_7d: Optional[float] = None
    apy_mean_30d: Optional[float] = None
    stablecoin: bool = False
    il_risk: str = "yes"              # "yes" / "no" per source
    exposure: str = "multi"           # "single" / "multi"
    outlier: bool = False
    age_days: Optional[int] = None    # number of daily observations at the source
    underlying_tokens: List[str] = field(default_factory=list)
    volume_usd_1d: Optional[float] = None
    volume_usd_7d: Optional[float] = None
    pool_meta: Optional[str] = None
    url: Optional[str] = None

    @property
    def tokens(self) -> List[str]:
        return [t for t in self.symbol.upper().split("-") if t]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Opportunity:
    """A pool after scoring: everything the manager needs to size and monitor it."""

    pool: Pool
    fee_apr: float          # decimal, conservative fee APR used for scoring
    reward_apr: float       # decimal, after haircut
    sigma: float            # annualised relative volatility of the pair
    vol_source: str         # "history" | "heuristic:<label>"
    amp: float              # IL / fee amplification vs full-range (1.0 = full range)
    fee_amp: float          # portion of amplification credited to fees
    range_width: Optional[float]
    expected_il_apr: float  # decimal, annualised expected IL
    cost_drag_apr: float    # decimal, entry+exit costs amortised over the horizon
    net_apr: float          # decimal
    breakeven_fee_apr: float
    confidence: float
    score: float
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        p = self.pool
        return (f"{p.chain:<10} {p.project:<24} {p.symbol:<22} tvl ${p.tvl_usd/1e6:7.1f}M "
                f"apy {p.apy:6.1f}%  fee {self.fee_apr*100:5.1f}%  rew {self.reward_apr*100:5.1f}%  "
                f"sigma {self.sigma:4.2f}  E[IL] {self.expected_il_apr*100:5.1f}%  "
                f"net {self.net_apr*100:6.1f}%  score {self.score*100:6.1f}")


@dataclass
class Position:
    position_id: str
    pool_id: str
    chain: str
    project: str
    symbol: str
    opened_at: str            # ISO-8601 UTC
    size_usd: float           # capital deployed at entry (after entry costs)
    entry_apy: float          # percent, total
    entry_fee_apr: float      # decimal
    entry_tvl: float
    entry_price_ratio: float  # price(tokenA)/price(tokenB) at entry, 1.0 when no IL exposure
    range_width: Optional[float]
    amp: float
    fee_amp: float
    sigma: float
    fees_accrued_usd: float = 0.0
    il_usd: float = 0.0       # negative when adverse
    value_usd: float = 0.0
    last_price_ratio: float = 1.0
    last_apy: float = 0.0
    last_fee_apr: float = 0.0
    last_tvl: float = 0.0
    last_net_apr: float = 0.0
    out_of_range_since: Optional[str] = None
    last_update: Optional[str] = None
    status: str = "open"      # open | closed
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None
    realised_pnl_usd: Optional[float] = None
    entry_cost_usd: float = 0.0
    exit_cost_usd: float = 0.0

    @property
    def net_pnl_usd(self) -> float:
        return self.fees_accrued_usd + self.il_usd

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Position":
        return cls(**d)


@dataclass
class Fill:
    ok: bool
    size_usd: float = 0.0     # capital actually placed (open) or returned (close)
    cost_usd: float = 0.0     # gas + slippage
    tx: Optional[str] = None
    note: str = ""
