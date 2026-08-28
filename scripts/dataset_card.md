---
license: apache-2.0
language:
  - en
tags:
  - bitcoin
  - cryptocurrency
  - finance
  - time-series
pretty_name: BERT-bitcoin-sentiment assets
---

# BERT-bitcoin-sentiment — large assets

Files that exceed GitHub's 100 MB limit, split out of the research repository at
<https://github.com/Kosmosas>. Fetch them into place with:

```bash
python scripts/download_assets.py
```

## Contents

| File | Size | What it is |
|---|---|---|
| `backtesting/data/BTCUSDT-last.csv` | ~411 MB | 1-minute BTCUSDT OHLCV + volume. The extended snapshot, running to Nov 2025; used by forecasting and backtesting. |
| `weights_comparison_and_derive/BTCUSDT.csv` | ~381 MB | 1-minute BTCUSDT OHLCV + volume. An earlier snapshot used by the headline-to-price-impact parameter study. |
| `fine_tuned_BERT/fine-tuned-finbert.pth` | ~439 MB | ⚠️ **Old version, and a duplicate.** See below. |

The two `BTCUSDT*.csv` files look alike but are different snapshots, with
different row counts and date ranges. Both are needed; they are not
interchangeable.

## About `fine_tuned_BERT/fine-tuned-finbert.pth`

This file is **superseded and does not belong in this dataset repo.** It is a
leftover copy from an earlier release layout, kept only so that existing links
and clones do not break. Nothing in the current code downloads it.

- The identical file lives in the model repo as
  [`Kosmosas/BERT-bitcoin-sentiment` → `fine-tuned-finbert.pth`](https://huggingface.co/Kosmosas/BERT-bitcoin-sentiment),
  which is the canonical location.
- More importantly, those are the **old** weights: a single checkpoint fine-tuned
  on 2019-01 → 2024-03 and then applied to its own training headlines, so its
  training-period output is an in-sample fitted value rather than a prediction.
- **For scoring or backtesting, use `oof/finbert_oof_tone3_cond_fold6.pth` in the
  model repo instead.** See that repo's card for the fold layout and calibration
  constants.

## License

Apache 2.0 for the derived files. The underlying market and news data carry their
own terms; check the original sources before redistributing.
