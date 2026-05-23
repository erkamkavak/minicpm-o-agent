# Duplex Prompt Update Benchmark Findings

This note summarizes the first real-model checks for mid-session duplex system
prompt updates. The benchmark compares two paths:

- **Update path:** start with the old prompt, replay history, update the system
  prompt in-place, then probe the next-token logits.
- **Full replay path:** start with the new prompt and replay the same raw
  history from scratch before probing the next-token logits.

The full replay path is treated as the reference because every historical unit is
encoded under the new prompt. The update path is faster because it rebuilds only
the protected system span and preserves later unit cache entries.

## Current Results

### Audio History, 30 Units

| Metric | Value |
|---|---:|
| Speedup | 68.5x |
| KL surgery -> full | 0.000822 |
| KL full -> surgery | 0.000656 |
| JS divergence | 0.000181 |
| Relative L2 logits | 0.0419 |
| Top-1 same | true |
| Top-5 overlap | 3 / 5 |
| Top-10 overlap | 7 / 10 |

This is a strong result for the cache mechanics. The distributions are very
close by KL/JS, the top token is unchanged, and the speedup is large.

### Text History, 30 Units, Short Prompt Change

| Metric | Value |
|---|---:|
| Speedup | 81.9x |
| KL surgery -> full | 0.0196 |
| KL full -> surgery | 0.0304 |
| JS divergence | 0.00579 |
| Relative L2 logits | 0.300 |
| Top-1 same | true |
| Top-5 overlap | 3 / 5 |
| Top-10 overlap | 6 / 10 |

This shows more drift than the audio fixture. The top token stayed the same, but
the larger KL/JS and smaller top-k overlap mean the preserved unit cache is no
longer a near-exact approximation of full replay.

### Text History, 30 Units, Longer Prompt Change

| Metric | Value |
|---|---:|
| Speedup | 71.7x |
| KL surgery -> full | 0.0119 |
| KL full -> surgery | 0.0135 |
| JS divergence | 0.00312 |
| Relative L2 logits | 0.110 |
| Top-1 same | true |
| Top-5 overlap | 4 / 5 |
| Top-10 overlap | 6 / 10 |

The longer prompt did not automatically make divergence worse. The amount of
drift depends on how the new prompt changes attention patterns, not only on
prompt length.

## Reading The Divergence Metrics

There is no universal KL-divergence threshold that proves two model states are
equivalent. The useful interpretation is relative and task-specific:

| Range | Practical meaning |
|---|---|
| `< 0.0001` | Extremely close for next-token distribution checks. |
| `0.0001 - 0.001` | Very close; usually acceptable for cache mechanics. |
| `0.001 - 0.01` | Small but visible drift; inspect top-k behavior. |
| `0.01 - 0.05` | Moderate drift; usable only if behavior checks still pass. |
| `> 0.05` | Significant drift; prefer hybrid replay or full replay. |

KL/JS should not be read alone. The most useful bundle is:

- KL in both directions, because KL is asymmetric.
- JS divergence, because it is symmetric and easier to compare across runs.
- Top-1 sameness, because this tells whether greedy decoding starts the same.
- Top-5 and top-10 overlap, because generation sampling can choose from this
  neighborhood.
- A short behavioral generation check, because logits are only a proxy.

## Hybrid Recent Replay

The current pure cache-surgery path preserves every historical unit. That is the
fastest path, but it is also the most approximate path because old units keep the
KV values they had under the old prompt.

The hybrid strategy is:

```text
old prompt + full history
drop the most recent N completed units from cache
update the protected system span
replay those N raw units under the new prompt
probe / continue generation
```

This keeps older cache entries for speed while refreshing the most recent
conversation turns, which are usually the most important for the next answer.

Expected tradeoff:

| Recache last N | Speed | Expected divergence |
|---:|---|---|
| 0 | fastest | highest approximation drift |
| 1-3 | still very fast | often improves next-token neighborhood |
| 5-10 | medium-fast | should reduce drift more noticeably |
| all | close to full replay | closest to reference |

The benchmark supports this with `MINICPMO45_RECACHE_LAST_UNITS`. Use `0`, `3`,
`5`, `10`, or `all` to sweep the curve.

## Recommendation

Use pure cache surgery when the new prompt is a small operational update, such as
adding or removing a tool. Use hybrid recent replay when the new prompt changes
assistant behavior, output format, or policy. Start by testing `3`, `5`, and `10`
recent units; pick the smallest value that keeps top-1 stable and brings JS/KL
into the range you are comfortable with.
