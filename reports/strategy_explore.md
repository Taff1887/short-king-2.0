# Strategy exploration -- 5 pre-specified variants

**Discipline:** all variants defined upfront in `scripts/_strategy_explore.py`. IS-only Sharpe used for variant selection. OOS Sharpe reported as unbiased confirmation of the IS-winner -- never used to retune.

## IS-only leaderboard (n=119 monthly OOF + 35 OOS = 154 ALL for trained models)

| variant | period | n_rebalances | avg_basket_size | Sharpe | CAGR | MaxDD | hit_rate |
|---|---|---|---|---|---|---|---|
| V3: naive+EW average L/S quintile | IS | 156 | 92.3 | 0.559 | 0.0898 | -0.4635 | 0.609 |
| V0: EW L/S quintile (baseline) | IS | 156 | 92.3 | 0.505 | 0.0866 | -0.5768 | 0.635 |
| V4: EW conviction-weighted positions | IS | 156 | 92.3 | 0.453 | 0.0818 | -0.6311 | 0.647 |
| V2: EW conviction gate (score > 0.85) | IS | 156 | 63.2 | 0.436 | 0.0785 | -0.5985 | 0.603 |
| V1: SI floor (>3%) + EW L/S quintile | IS | 149 | 19.6 | 0.33 | 0.0524 | -0.6154 | 0.577 |
| V0b: EW quintile-short only | IS | 156 | 46.2 | -0.221 | -0.1066 | -0.8608 | 0.449 |

## OOS confirmation (n=35)

| variant | n_rebalances | avg_basket_size | Sharpe | CAGR | MaxDD | hit_rate |
|---|---|---|---|---|---|---|
| V0: EW L/S quintile (baseline) | 35 | 92.3 | 0.909 | 0.1707 | -0.1657 | 0.486 |
| V0b: EW quintile-short only | 35 | 46.2 | -0.077 | -0.0403 | -0.3646 | 0.514 |
| V1: SI floor (>3%) + EW L/S quintile | 35 | 19.6 | 0.967 | 0.2456 | -0.1655 | 0.514 |
| V2: EW conviction gate (score > 0.85) | 35 | 63.2 | 0.849 | 0.1764 | -0.1656 | 0.543 |
| V3: naive+EW average L/S quintile | 35 | 92.3 | 1.453 | 0.2383 | -0.1101 | 0.714 |
| V4: EW conviction-weighted positions | 35 | 92.3 | 0.988 | 0.2146 | -0.1522 | 0.571 |

## All variants, all periods

| variant | period | n_rebalances | avg_basket_size | Sharpe | CAGR | MaxDD | hit_rate |
|---|---|---|---|---|---|---|---|
| V0: EW L/S quintile (baseline) | ALL | 191 | 92.3 | 0.575 | 0.1015 | -0.5768 | 0.607 |
| V0: EW L/S quintile (baseline) | IS | 156 | 92.3 | 0.505 | 0.0866 | -0.5768 | 0.635 |
| V0: EW L/S quintile (baseline) | OOS | 35 | 92.3 | 0.909 | 0.1707 | -0.1657 | 0.486 |
| V0b: EW quintile-short only | ALL | 191 | 46.2 | -0.199 | -0.0948 | -0.8608 | 0.461 |
| V0b: EW quintile-short only | IS | 156 | 46.2 | -0.221 | -0.1066 | -0.8608 | 0.449 |
| V0b: EW quintile-short only | OOS | 35 | 46.2 | -0.077 | -0.0403 | -0.3646 | 0.514 |
| V1: SI floor (>3%) + EW L/S quintile | ALL | 184 | 19.6 | 0.442 | 0.0867 | -0.6154 | 0.565 |
| V1: SI floor (>3%) + EW L/S quintile | IS | 149 | 19.6 | 0.33 | 0.0524 | -0.6154 | 0.577 |
| V1: SI floor (>3%) + EW L/S quintile | OOS | 35 | 19.6 | 0.967 | 0.2456 | -0.1655 | 0.514 |
| V2: EW conviction gate (score > 0.85) | ALL | 191 | 63.2 | 0.506 | 0.0958 | -0.5985 | 0.592 |
| V2: EW conviction gate (score > 0.85) | IS | 156 | 63.2 | 0.436 | 0.0785 | -0.5985 | 0.603 |
| V2: EW conviction gate (score > 0.85) | OOS | 35 | 63.2 | 0.849 | 0.1764 | -0.1656 | 0.543 |
| V3: naive+EW average L/S quintile | ALL | 191 | 92.3 | 0.699 | 0.1156 | -0.4635 | 0.628 |
| V3: naive+EW average L/S quintile | IS | 156 | 92.3 | 0.559 | 0.0898 | -0.4635 | 0.609 |
| V3: naive+EW average L/S quintile | OOS | 35 | 92.3 | 1.453 | 0.2383 | -0.1101 | 0.714 |
| V4: EW conviction-weighted positions | ALL | 191 | 92.3 | 0.545 | 0.105 | -0.6311 | 0.634 |
| V4: EW conviction-weighted positions | IS | 156 | 92.3 | 0.453 | 0.0818 | -0.6311 | 0.647 |
| V4: EW conviction-weighted positions | OOS | 35 | 92.3 | 0.988 | 0.2146 | -0.1522 | 0.571 |