# Crypto-LP-Strategy
Experimental. **Paper trading is complete; live execution is an interface you must implement
and test per protocol before real funds touch it.** Not financial advice.

## What it does

1. **Scan every chain.** Pulls every pool DefiLlama's yields API tracks (thousands of pools
   across Ethereum, L2s, Solana, BSC, Avalanche, Sui, Aptos, and more, on every DEX it indexes)
   and normalises them into one `Pool` record. Add other venues by implementing `PoolSource`.
2. **Filter out traps.** TVL floor, pool age, outlier flag, implausible APY, emission-heavy
   APY, single-asset vaults (not LPing), turnover too low for the fee number to be real, chain,
   protocol and token block lists. Perpetual-futures liquidity pools (GMX, Gains, Jupiter
   perps and similar) and tokenised-stock pairs are blocked by default because the IL model does
   not describe their risk; your config's lists add to those defaults.
3. **Score net of impermanent loss.** For each survivor it estimates the volatility of the
   pair's price ratio (realised from 90 days of price history when available, class-based
   heuristics otherwise) and computes

   ```
   net_apr = fee_amp * fee_apr + haircut * reward_apr - amp * sigma^2 / 8 - cost_drag
   ```

   `amp * sigma^2 / 8` is the annualised expected IL for a lognormal price ratio; `amp` is the
   amplification of a symmetric concentrated-liquidity range (1.0 for full range). Fee APR uses
   the slower of the 1-day and 7-day figures and is capped at 1.5x the 30-day mean, emissions
   are valued at 50 cents on the dollar, and gas plus slippage are amortised over the holding
   horizon. Every rejection carries a reason (`scan --explain`).
4. **Manage positions in USD.** Each position records the USD prices of both tokens at entry,
   so every hourly mark splits its result into four parts:

   * **market**: what simply holding the two tokens would have made or lost;
   * **IL**: the LP's shortfall versus holding (amplified for concentrated ranges);
   * **fees**: accrued at the pool's current fee rate while in range;
   * **rewards**: emissions at the same 50 % haircut used when choosing the pool.

   A position is never opened unless both tokens can be priced, and never when a stablecoin
   leg is already off its peg. Prices come from contract addresses, falling back to CoinGecko
   ids. Each cycle first marks every position and runs the hard exits, and only then does
   the slow work of ranking new pools.

   | Rule | Default | Type |
   |---|---|---|
   | Kill switch file present (works even when the feed is down) | `lp_bot.STOP` | hard, closes everything |
   | Fees + rewards + IL below `-il_stop_pct` of size | 3 % | hard |
   | A USD stablecoin leg trades off its peg (yield-bearing ones: falls below entry) | 2 % | hard |
   | A tightly pegged pair (LST/ETH, BTC wrappers) diverges beyond its expected drift | 2 % | hard |
   | Out of range for longer than the grace period | 12 h | hard |
   | Pool missing from an otherwise healthy feed | 24 h | hard |
   | Position cannot be priced while others can | 6 h | hard |
   | Pool excluded by a block list you added | — | hard |
   | Optional absolute price-move stop | off | hard |
   | Expected to lose money from here (fees + rewards below expected IL) | `min_hold_apr` 0 % | soft, after `min_hold_hours` |
   | 7-day fee rate below half of the rate bought on, and the pool would not be bought today | 0.5x | soft |
   | Price-adjusted TVL below half of entry TVL | 0.5x | soft |
   | Held `max_hold_days` and would not be bought today | 90 d | soft |
   | Rotation: a better pool nets +8 pp over the weakest holding | 0.08 | soft, one per cycle |

   Soft exits judge the position on its forward economics, using its own range model: the
   smoothed 7-day fee rate and rewards against expected IL. Costs are left out of that test:
   the entry cost is already spent and the exit cost is paid whether the bot sells now or
   later. One quiet day or a spike fading back to normal is not a reason to sell. Fees are
   booked on the liquidity's current value, and a single implausible fee print is capped at
   3x the 7-day level so it cannot hide a real IL loss.

   Entries are funded from free cash and free slots first. Rotation then pairs the best
   candidate that could not be funded with the weakest holding, and only when the candidate
   can actually be sized and priced. The daily loss limit blocks entries and rotations. A pool
   that just exited cannot be re-entered for 24 h, and a pair that hit a hard stop cannot be
   bought in any pool for 24 h. When the yields feed or the price feed is down, the bot holds,
   runs the price-based stops, and buys nothing.

   If the bot was not running, the first mark books the gap conservatively: the lower of the
   before and after fee rates, and only if the position was in range at both ends.
5. **Execute.** `PaperExecutor` simulates mints and burns with per-chain gas, slippage (lower
   for pegged pairs), and a bridge hop for any chain other than `home_chain`. `LiveExecutor`
   routes each protocol to a registered `ProtocolAdapter` and refuses to start without an
   explicit environment acknowledgement, a signer, and at least one adapter.

## Running it safely

* Only one `run` can use a state file at a time; a second one exits with an error. `status`
  is read-only and safe to run alongside the loop.
* State is saved after each phase of a cycle with an fsync, the previous file is kept as
  `lp_bot_state.json.bak`, and a corrupt main file falls back to it automatically.
* An error inside a cycle is printed and the loop carries on; the yields feed is retried
  three times before the cycle holds.
* State files from older versions load as-is. Their positions switch to USD valuation on the
  first mark: a stablecoin leg anchors the entry at $1, otherwise historical prices at the
  entry time are used, and a stored placeholder ratio is replaced by the real one. The
  one-off revaluation resets the daily-loss reference, and notes in `status` say what was
  recovered. Delete `lp_bot_state.json` to start a fresh paper account.

## Usage

```
python3 -m lp_bot self-test                              # deterministic checks, no network
python3 -m lp_bot scan --source fixture --prices none    # offline demo on synthetic data
python3 -m lp_bot scan --top 25 --explain                # live scan, all chains
python3 -m lp_bot scan --chains base arbitrum solana --json top.json
python3 -m lp_bot run --config my.json --once            # one paper cycle, persists state
python3 -m lp_bot run --config my.json --interval-min 60 # loop
python3 -m lp_bot status
python3 -m lp_bot il --k 1.5 --range 3 --sigma 0.7 --budget 0.03
python3 -m lp_bot prices coingecko:ethereum coingecko:tether   # debug the price feed
```

On Windows use `python` instead of `python3` and `copy` instead of `cp`.

Copy `lp_bot/config.example.json`, edit, and pass it with `--config`. State lives in
`lp_bot_state.json` (git-ignored). Touch `lp_bot.STOP` to close everything and halt.

## Going live

Nothing in this repository signs a transaction. To trade for real you need, per protocol
family you want to use, a `ProtocolAdapter` (see `executors.py`) that:

* builds a wallet from `LP_BOT_PRIVATE_KEY` or a remote/hardware signer and never logs it;
* swaps the funding asset into both legs, approves, mints the position (for concentrated
  pools, a symmetric range of `opp.range_width` around the current price), and reports the
  capital actually placed;
* burns, collects fees, swaps back, and reports the USD realised;
* enforces on-chain slippage limits and, ideally, replaces the modelled `fees_accrued_usd`
  with fees read from the chain.

Register it with `register_adapter("uniswap-v3", factory)`, set `mode` to `live`, export
`LP_BOT_LIVE_ACK=I_UNDERSTAND_THIS_TRADES_REAL_FUNDS`, and validate on a testnet or with dust.
This will need `web3.py` (EVM) or `solders`/`solana-py` (Solana); the core stays stdlib-only.
In live mode the planner only enters pools whose protocol has an adapter.

## Known limitations

* Fees are modelled from the source's fee APY, not read on-chain, until an adapter does so.
* Expected IL uses the second-order lognormal approximation; it understates very large moves.
  The realised-IL stop, not the estimate, is what protects a position.
* Volatile concentrated pools are modelled as symmetric ranges of `default_range_width`. That
  amplifies IL, but fees are credited at the pool-average APY (`cl_fee_amp_share` = 0) because
  DefiLlama's pool-level number does not say what a specific range earns. A wide range
  usually earns less than the pool average, so paper fees on concentrated pools are an upper
  bound. Pegged pairs (sigma below `stable_sigma_threshold`) in concentrated pools are
  modelled as a tight range around the peg, widened by the pair's measured drift so a
  yield-bearing leg does not walk it out of range. Stableswap pools such as Curve have no
  range; there the de-peg and divergence stops do the protecting.
* Bridging is a flat cost per hop from `home_chain`; there is no bridge delay and no per-chain
  gas inventory, so cross-chain paper results are still a little optimistic.
* The turnover filter (`min_volume_tvl_ratio_7d`) is off by default and does not check the
  fee figure against volume.
* Three-plus-token pools are reduced to their most volatile pair for IL purposes.
* Volatility falls back to class heuristics when price history cannot be fetched; the scan
  output says which was used.
* Costs are amortised over `horizon_days` at the size the bot would actually buy (20 % of
  equity by default). Mainnet gas on a $5k position is a large drag, which is why most
  Ethereum pools score below the floor at that size. Raise the size, lengthen the horizon, or
  use L2s.
