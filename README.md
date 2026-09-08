# Crypto-LP-Strategy
Experimental. **Paper trading is complete; live execution is an interface you must implement
and test per protocol before real funds touch it.** Not financial advice.

## What it does

1. **Scan every chain.** Pulls every pool DefiLlama's yields API tracks (thousands of pools
   across Ethereum, L2s, Solana, BSC, Avalanche, Sui, Aptos, and more, on every DEX it indexes)
   and normalises them into one `Pool` record. Add other venues by implementing `PoolSource`.
2. **Filter out traps.** TVL floor, pool age, outlier flag, implausible APY, emission-heavy
   APY, single-asset vaults (not LPing), turnover too low for the fee number to be real, chain,
   protocol and token block lists.
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
4. **Manage positions.** Each cycle it marks every open position (fees accrued from the
   pool's fee APR, IL from the price ratio move since entry, out-of-range detection for
   concentrated positions) and applies exit rules:

   | Rule | Default | Type |
   |---|---|---|
   | Fees + IL below `-il_stop_pct` of size | 3 % | hard, immediate |
   | Out of range for longer than the grace period | 12 h | hard |
   | Optional absolute price-move stop | off | hard |
   | Pool vanished from the feed or flagged outlier | — | hard |
   | Kill switch file present | `lp_bot.STOP` | hard, closes everything |
   | APY fell below half of entry APY | 0.5x | soft, after `min_hold_hours` |
   | TVL fell below half of entry TVL | 0.5x | soft |
   | Held longer than `max_hold_days` | 90 d | soft |
   | A better pool nets +8 pp after switching costs | 0.08 | rotation, after min hold |

   Entries respect caps per pool, protocol, and chain, a cash reserve, a daily loss limit that
   halts new entries, and a cooldown that blocks re-entering a pool that just stopped out.
   Small positions are re-scored at their real size so gas cannot quietly eat them.
5. **Execute.** `PaperExecutor` simulates mints and burns with per-chain gas and slippage.
   `LiveExecutor` routes each protocol to a registered `ProtocolAdapter` and refuses to start
   without an explicit environment acknowledgement, a signer, and at least one adapter.

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
* Concentrated pools are modelled as symmetric ranges; DefiLlama's pool-level fee APY does not
  say what a specific range earns, so only half the range amplification is credited to fees
  (`cl_fee_amp_share`). Tune it against your own fills.
* Three-plus-token pools are reduced to their most volatile pair for IL purposes.
* Volatility falls back to class heuristics when price history cannot be fetched; the scan
  output says which was used.
* Mainnet gas plus slippage on a $10k position amortised over 30 days is roughly 15 %/yr of
  drag, which is why most Ethereum stable pools score below the floor at that size. Raise the
  size, lengthen the horizon, or use L2s.
