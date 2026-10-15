"""Configuration with conservative defaults. Override with a JSON file (see config.example.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


DEFAULT_GAS_ROUNDTRIP_USD: Dict[str, float] = {
    # Rough cost of mint + burn (+ swaps to balance) in USD. Refresh periodically.
    "ethereum": 60.0,
    "arbitrum": 1.5,
    "optimism": 1.0,
    "base": 1.0,
    "polygon": 0.5,
    "bsc": 1.5,
    "avalanche": 2.0,
    "solana": 0.5,
    "sui": 0.5,
    "aptos": 0.5,
    "default": 3.0,
}


# Perpetual-futures liquidity pools: the LP is the counterparty to leveraged traders, which
# the impermanent-loss model does not describe. Blocked by default; edit in your config.
DEFAULT_PROJECT_BLOCKLIST = [
    "gmx", "gmx-v2", "gmx-v2-perps", "gmx-solana", "gains-network", "jupiter-perpetual-exchange",
    "hyperliquid-hlp", "vertex-protocol", "mux-protocol", "kwenta", "level-finance", "flash-trade",
]
# Tokenised stocks trade 24/7 while the underlying market is closed; LPs get arbitraged at the
# open and the measured volatility understates the gap risk.
DEFAULT_TOKEN_BLOCKLIST = [
    "CRCLX", "NVDAX", "SPYX", "TSLAX", "AAPLX", "MSTRX", "GOOGLX", "AMZNX", "METAX", "COINX",
    "HOODX", "MSFTX", "QQQX", "GMEX", "NFLXX", "AVGOX", "PLTRX", "LLYX", "JPMX", "NKE", "SPYX",
    "NVDAX", "CRCLX", "TQQQX", "USDCAD", "USDJPY", "EURUSD", "GBPUSD",
]


@dataclass
class Config:
    # --- capital & mode ------------------------------------------------------------
    mode: str = "paper"                    # "paper" | "live"
    starting_cash_usd: float = 25_000.0    # paper mode only
    state_path: str = "lp_bot_state.json"
    kill_switch_path: str = "lp_bot.STOP"  # if this file exists, exit everything and stop

    # --- universe -----------------------------------------------------------------
    chains: List[str] = field(default_factory=list)          # [] = every chain
    chain_blocklist: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)        # [] = every protocol
    project_blocklist: List[str] = field(default_factory=lambda: list(DEFAULT_PROJECT_BLOCKLIST))
    token_blocklist: List[str] = field(default_factory=lambda: list(DEFAULT_TOKEN_BLOCKLIST))
    require_multi_asset: bool = True       # skip single-sided vaults/lending (not LPing)
    allow_reward_heavy: bool = False

    # --- pool quality filters --------------------------------------------------------
    min_tvl_usd: float = 2_000_000.0
    min_pool_age_days: int = 30
    min_apy_pct: float = 5.0
    max_apy_pct: float = 250.0             # above this the number is almost always a trap
    max_reward_share: float = 0.6          # emissions / total APY
    min_volume_tvl_ratio_7d: float = 0.0   # 0 disables; 0.05 means >=5 % weekly turnover
    reject_outliers: bool = True           # entry filter only; the flag is not an exit signal
    allow_unknown_age: bool = False        # pools with no age data are rejected by default

    # --- scoring --------------------------------------------------------------------
    reward_haircut: float = 0.5            # value emissions at 50 cents on the dollar
    horizon_days: int = 30                 # holding period used to amortise costs
    default_range_width: float = 3.0       # symmetric CL range [P/3, P*3]; None = full range
    cl_fee_amp_share: float = 0.0          # fraction of IL amplification credited to fees (0 = pool average)
    stable_sigma_threshold: float = 0.15   # below this the pair is treated as a stable/pegged pair
    stable_range_sigmas: float = 2.0       # pegged pairs in CL pools: range half-width in horizon sigmas (min 1 %)
    peg_pair_sigma: float = 0.05           # pairs this tightly correlated get a divergence stop
    max_sigma: float = 1.5                 # skip pairs more volatile than this (annualised)
    max_expected_il_apr: float = 0.35      # skip when expected IL alone exceeds 35 %/yr
    min_net_apr: float = 0.04              # 4 % net after IL and costs
    slippage_bps_per_side: float = 30.0
    stable_slippage_bps_per_side: float = 5.0   # pegged pairs swap at a fraction of the cost
    gas_roundtrip_usd: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GAS_ROUNDTRIP_USD))
    home_chain: str = "arbitrum"           # where idle paper cash is assumed to sit
    bridge_cost_usd: float = 3.0           # fixed cost per bridge hop to or from another chain
    bridge_bps: float = 5.0                # plus this share of the amount bridged

    # --- sizing ---------------------------------------------------------------------
    max_positions: int = 8
    min_position_usd: float = 500.0
    max_position_usd: float = 10_000.0
    max_pct_per_pool: float = 0.20
    max_pct_per_project: float = 0.35
    max_pct_per_chain: float = 0.50
    max_share_of_pool_tvl: float = 0.005   # never be more than 0.5 % of a pool
    cash_reserve_pct: float = 0.05

    # --- position management ----------------------------------------------------------
    il_stop_pct: float = 0.03              # exit when fees + IL < -3 % of size
    price_move_stop: float = 0.0           # 0 disables; else exit when |ln k| exceeds this
    out_of_range_grace_hours: float = 12.0
    apy_decay_ratio: float = 0.5           # exit when smoothed fee APR < 50 % of the fee APR bought on (0 = off)
    tvl_drop_ratio: float = 0.5            # exit when price-adjusted TVL < 50 % of entry TVL
    min_hold_hours: float = 48.0           # soft exits wait this long (costs must be earned back)
    min_hold_apr: float = 0.0              # keep a position while its forward yield (no costs) is above this
    max_hold_days: int = 90
    rotation_margin_apr: float = 0.08      # need +8 pp net APR to justify switching
    max_rotations_per_cycle: int = 1
    reentry_cooldown_hours: float = 24.0   # do not re-enter a pool we just exited
    daily_loss_limit_pct: float = 0.05     # stop opening new positions after a -5 % day
    depeg_stop_pct: float = 0.02           # exit when a USD stablecoin leg trades 2 % off $1
    missing_grace_hours: float = 24.0      # exit when a pool is absent from the feed this long
    price_stale_hours: float = 6.0         # exit when a position cannot be priced this long
    gap_hours: float = 3.0                 # marks further apart than this are booked conservatively

    # --- data ----------------------------------------------------------------------------
    yields_url: str = "https://yields.llama.fi/pools"
    prices_url: str = "https://coins.llama.fi"
    price_history_days: int = 90
    http_timeout_s: int = 30
    fetch_retries: int = 3                 # attempts at the yields feed per cycle
    min_feed_pools: int = 1000             # a live feed smaller than this is treated as down
    prefetch_budget_s: float = 120.0       # max time per cycle spent fetching price history
    vol_refresh_hours: float = 24.0        # refresh price history (and so volatility) this often
    price_retry_hours: float = 6.0         # retry coins that returned no history after this long

    def gas_for(self, chain: str) -> float:
        return self.gas_roundtrip_usd.get(chain.lower(), self.gas_roundtrip_usd.get("default", 3.0))

    def bridge_for(self, chain: str, amount_usd: float) -> float:
        """One-way cost of moving funds between the home chain and ``chain``."""
        if chain.strip().lower() == self.home_chain.strip().lower():
            return 0.0
        return self.bridge_cost_usd + max(amount_usd, 0.0) * self.bridge_bps / 10_000.0

    def one_way_cost(self, chain: str, amount_usd: float, stable: bool = False) -> float:
        """Gas for one leg (mint or burn), slippage on one side, and one bridge hop."""
        bps = self.stable_slippage_bps_per_side if stable else self.slippage_bps_per_side
        return (self.gas_for(chain) / 2.0
                + max(amount_usd, 0.0) * bps / 10_000.0
                + self.bridge_for(chain, amount_usd))

    def to_dict(self) -> Dict:
        return asdict(self)


def load_config(path: Optional[str]) -> Config:
    cfg = Config()
    if not path:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = [k for k in raw if k not in cfg.__dict__]
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    for k, v in raw.items():
        if k in ("project_blocklist", "token_blocklist"):
            v = list(dict.fromkeys(list(getattr(cfg, k)) + [str(x) for x in v]))
        if k == "gas_roundtrip_usd":
            merged = dict(DEFAULT_GAS_ROUNDTRIP_USD)
            merged.update({kk.lower(): float(vv) for kk, vv in v.items()})
            v = merged
        setattr(cfg, k, v)
    if cfg.mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    if cfg.default_range_width is not None and cfg.default_range_width <= 1.0:
        raise ValueError("default_range_width must be > 1 or null")
    return cfg
