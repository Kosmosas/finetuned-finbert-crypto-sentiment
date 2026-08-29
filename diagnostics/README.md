# Diagnostics

Read-only scripts that re-derive, from the code and the saved checkpoints, the
methodological facts the manuscript asserts. None of them train anything; each
prints its findings and exits. Run from the repository root.

| Script | Establishes | Runtime |
|---|---|---|
| `finbert_head_audit.py` | Which parameters of the headline scoring model are trainable: the `Linear(3,1)` head is frozen at its random initialisation, the pretrained tone classifier is frozen at its pretrained values, and only encoder blocks 10–11 receive gradients. Contrasts the frozen head against the trained `Linear(768,1)` control. | ~1 min |
| `normalization_audit.py` | The two-stage normalisation of the forecasting inputs — an external per-variate z-score fitted on training windows only, plus RevIN inside the iTransformer — and measures why both are needed via RevIN's `eps` variance floor. | ~3 min |
| `training_data_audit.py` | The exact training interval and sequence counts per feature set, and every hour on which two feature sets disagree. | ~2 min |
| `backtest_forecast_audit.py` | What share of the strategy-parameter search period the forecaster had already been fitted on, i.e. how much of the tuning signal is in-sample. | ~2 min |

```bash
python diagnostics/finbert_head_audit.py
python diagnostics/normalization_audit.py
python diagnostics/training_data_audit.py
python diagnostics/backtest_forecast_audit.py
```

`finbert_head_audit.py` needs the out-of-fold checkpoints. If they are not present
locally it downloads them from the Hugging Face model repo; see
`scripts/download_assets.py --all`.

The remaining scripts read the hourly feature sets, which are cached under
`lstm_models/cache/` after the first run of any `01_*` notebook. Without that
cache the first one pays a few minutes to parse `BTCUSDT-last.csv`.

Two related pieces of code live with the stage they belong to rather than here,
because they are not read-only:

- `itransformer_models/matched_training_check.py` — retrains Price + volume and
  Price + volume + sentiment on the intersection of their hourly indices, so the
  two see identical rows. ~1.5 min on a 12 GB GPU.
- `results/make_dm_tables.py` — rewrites both Diebold–Mariano tables in one
  uniform format (mean loss differential, DM, raw p, BH-adjusted q).
