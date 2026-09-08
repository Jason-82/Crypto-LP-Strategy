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
    project_blocklist: List[str] = field(default_factory=list)
    token_blocklist: List[str] = field(default_factory=list)  # symbols never to touch
    require_multi_asset: bool = True       # skip single-sided vaults/lending (not LPing)
    allow_reward_heavy: bool = False

    # --- pool quality filters --------------------------------------------------------
    min_tvl_usd: float = 2_000_000.0
    min_pool_age_days: int = 30
    min_apy_pct: float = 5.0
    max_apy_pct: float = 250.0             # above this the number is almost always a trap
    max_reward_share: float = 0.6          # emissions / total APY
    min_volume_tvl_ratio_7d: float = 0.0   # 0 disables; 0.05 means >=5 % weekly turnover
    reject_outliers: bool = True

    # --- scoring --------------------------------------------------------------------
    reward_haircut: float = 0.5            # value emissions at 50 cents on the dollar
    horizon_days: int = 30                 # holding period used to amortise costs
    default_range_width: float = 3.0       # symmetric CL range [P/3, P*3]; None = full range
    cl_fee_amp_share: float = 0.5          # fraction of IL amplification credited to fees
    max_sigma: float = 1.5                 # skip pairs more volatile than this (annualised)
    max_expected_il_apr: float = 0.35      # skip when expected IL alone exceeds 35 %/yr
    min_net_apr: float = 0.04              # 4 % net after IL and costs
    slippage_bps_per_side: float = 30.0
    gas_roundtrip_usd: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GAS_ROUNDTRIP_USD))

    # --- sizing ---------------------------------------------------------------------
    max_positions: int = 8
    min_position_usd: float = 500.0
    max_position_usd: float = 10_000.0
    max_pct_per_pool: float = 0.20
    max_pct_per_project: float = 0.35
    max_pct_per_chain: float = 0.50
    cash_reserve_pct: float = 0.05

    # --- position management ----------------------------------------------------------
    il_stop_pct: float = 0.03              # exit when fees + IL < -3 % of size
    price_move_stop: float = 0.0           # 0 disables; else exit when |ln k| exceeds this
    out_of_range_grace_hours: float = 12.0
    apy_decay_ratio: float = 0.5           # exit when current APY < 50 % of entry APY
    tvl_drop_ratio: float = 0.5            # exit when TVL < 50 % of entry TVL
    min_hold_hours: float = 48.0           # soft exits wait this long (costs must be earned back)
    max_hold_days: int = 90
    rotation_margin_apr: float = 0.08      # need +8 pp net APR to justify switching
    reentry_cooldown_hours: float = 24.0   # do not re-enter a pool we just exited
    daily_loss_limit_pct: float = 0.05     # stop opening new positions after a -5 % day

    # --- data ----------------------------------------------------------------------------
    yields_url: str = "https://yields.llama.fi/pools"
    prices_url: str = "https://coins.llama.fi"
    price_history_days: int = 90
    http_timeout_s: int = 30

    def gas_for(self, chain: str) -> float:
        return self.gas_roundtrip_usd.get(chain.lower(), self.gas_roundtrip_usd.get("default", 3.0))

    def to_dict(self) -> Dict:
        return asdict(self)


def load_config(path: Optional[str]) -> Config:
    cfg = Config()
    if not path:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    unknown = [k for k in raw if k not in cfg.__dict__]
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    for k, v in raw.items():
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
