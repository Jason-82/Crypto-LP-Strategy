"""Impermanent-loss arithmetic for constant-product and concentrated-liquidity positions.

Conventions
-----------
* ``k`` is the price ratio change since entry: ``k = (Pa/Pb)_now / (Pa/Pb)_entry``.
* Losses are returned as negative decimals (``-0.057`` means -5.7 % versus holding).
* A concentrated position is modelled as a symmetric range ``[P/r, P*r]`` around the
  entry price with width ``r > 1``. ``r = None`` means full range (constant product).
* ``sigma`` is the annualised volatility of ``ln(Pa/Pb)``.

The concentrated model follows the standard result that, while in range, a symmetric
Uniswap-v3 style position experiences IL (and earns fees) amplified by
``sqrt(r) / (sqrt(r) - 1)`` relative to a full-range position. Once out of range the
position is entirely one asset and drifts linearly with price. This is an
approximation that ignores fee compounding and discrete tick spacing.
"""

from __future__ import annotations

import math
from typing import Optional


def hold_value(k: float) -> float:
    """Value of holding a 50/50 basket, normalised to 1.0 at entry, in quote terms."""
    return (1.0 + k) / 2.0


def il_constant_product(k: float) -> float:
    """IL of a 50/50 constant-product position versus holding. Always <= 0."""
    if k <= 0:
        raise ValueError("price ratio must be positive")
    return 2.0 * math.sqrt(k) / (1.0 + k) - 1.0


def amplification(range_width: Optional[float]) -> float:
    """Fee/IL amplification of a symmetric range of width ``r`` versus full range."""
    if range_width is None or range_width == math.inf:
        return 1.0
    if range_width <= 1.0:
        raise ValueError("range_width must be > 1 (e.g. 1.5 means [P/1.5, P*1.5])")
    s = math.sqrt(range_width)
    return s / (s - 1.0)


def position_value(k: float, range_width: Optional[float]) -> float:
    """Value of the LP position (normalised to 1.0 at entry, quote terms) at ratio ``k``."""
    amp = amplification(range_width)
    if range_width is None:
        return hold_value(k) * (1.0 + il_constant_product(k))
    lo, hi = 1.0 / range_width, range_width
    if lo <= k <= hi:
        return hold_value(k) * (1.0 + amp * il_constant_product(k))
    if k > hi:
        # price rose through the top of the range: position is 100 % quote, value frozen
        return hold_value(hi) * (1.0 + amp * il_constant_product(hi))
    # price fell through the bottom: position is 100 % base and tracks the price
    v_lo = hold_value(lo) * (1.0 + amp * il_constant_product(lo))
    return v_lo * (k / lo)


def impermanent_loss(k: float, range_width: Optional[float] = None) -> float:
    """IL versus holding for a full-range (None) or symmetric concentrated position."""
    if k <= 0:
        raise ValueError("price ratio must be positive")
    if range_width is None:
        return il_constant_product(k)
    return position_value(k, range_width) / hold_value(k) - 1.0


def in_range(k: float, range_width: Optional[float]) -> bool:
    if range_width is None:
        return True
    return 1.0 / range_width <= k <= range_width


def expected_il_rate(sigma: float, amp: float = 1.0) -> float:
    """Annualised expected IL (positive number = expected drag) for lognormal prices.

    Second-order expansion: IL ~= -(ln k)^2 / 8, and E[(ln k)^2] = sigma^2 * T, so the
    expected drag per year is ``amp * sigma^2 / 8``. It is a decent estimate for
    moderate moves and understates the loss for very large moves.
    """
    return amp * sigma * sigma / 8.0


def breakeven_fee_apr(sigma: float, amp: float = 1.0, fee_amp: float = 1.0) -> float:
    """Full-range-equivalent fee APR needed for fees to cover expected IL."""
    return expected_il_rate(sigma, amp) / fee_amp


def pair_sigma(sigma_a: float, sigma_b: float, rho: float) -> float:
    """Volatility of ln(Pa/Pb) from the two legs' vols and their correlation."""
    var = sigma_a ** 2 + sigma_b ** 2 - 2.0 * rho * sigma_a * sigma_b
    return math.sqrt(max(var, 0.0))


def price_move_for_il(target_il: float, range_width: Optional[float] = None) -> float:
    """Smallest upward ratio k >= 1 at which IL reaches ``target_il`` (negative decimal).

    Bisection; useful for translating an IL budget into a price-move stop.
    """
    if target_il >= 0:
        return 1.0
    lo, hi = 1.0, 1.0
    while impermanent_loss(hi, range_width) > target_il:
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if impermanent_loss(mid, range_width) > target_il:
            lo = mid
        else:
            hi = mid
    return hi
