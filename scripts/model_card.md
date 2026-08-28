---
license: apache-2.0
language:
  - en
base_model: yiyanghkust/finbert-tone
pipeline_tag: text-classification
tags:
  - finbert
  - sentiment-analysis
  - finance
  - bitcoin
  - cryptocurrency
  - regression
---

# BERT-bitcoin-sentiment

FinBERT fine-tuned to predict the short-horizon Bitcoin price impact of a news
headline, as a continuous score rather than a class label. Released with the
accompanying paper; code at <https://github.com/Kosmosas>.

## Which file do I want?

**Use `oof/finbert_oof_tone3_cond_fold6.pth`.** It is the checkpoint that scores
the paper's evaluation window and the one every downstream forecasting and
backtesting result is built on. If you just want to score new headlines, that is
the file.

The repository holds two generations of weights:

| File | Status |
|---|---|
| `oof/finbert_oof_tone3_cond_fold{1..6}.pth` | **Current.** Expanding-window out-of-fold set. Fold 6 is the scoring model. |
| `fine-tuned-finbert.pth` | **Superseded.** Kept for reference and reproducibility of the in-sample comparison reported in the paper; not recommended for new work. |

The older checkpoint was fine-tuned once on 2019-01 → 2024-03 and then applied to
those same headlines, so over the training period its output is an in-sample
fitted value, not a prediction (correlation 0.477 in-sample against 0.071 out of
sample, with a 4.7x train/test standard-deviation break). The out-of-fold set
exists to remove that: each fold is trained only on strictly earlier data, with a
one-day purge around every boundary.

## The folds

Each checkpoint scores its own time slice and was trained on everything before it.

| Checkpoint | Scores | Train n | Out-of-fold corr. |
|---|---|---|---|
| `fold1` | 2020-01 → 2021-01 | 2,261 | −0.032 |
| `fold2` | 2021-01 → 2022-01 | 4,567 | +0.018 |
| `fold3` | 2022-01 → 2023-01 | 7,089 | −0.012 |
| `fold4` | 2023-01 → 2024-01 | 12,457 | +0.034 |
| `fold5` | 2024-01 → 2024-06 | 15,180 | +0.127 |
| **`fold6`** | **2024-06 → 2025-07** | **16,187** | **+0.076** |

Correlation is against `volume_surge_price`, the composite price-impact target.
Nothing before 2020-01 has a score: there is no earlier data to train the first
fold on, and that gap is left as `NaN` rather than filled with zero.

## Architecture

`BertForSequenceClassification` (`num_labels=3`) with a `Linear(3, 1)` head on the
three tone logits. The last two encoder blocks are trainable (14.18M of 109.8M
parameters); the rest of the backbone is frozen.

Two differences from the superseded checkpoint matter when loading:

- **No `tanh`.** The output is linear. Scores are not bounded to `[-1, 1]`.
- **Max sequence length 128**, not 256.

## Usage

```python
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import BertForSequenceClassification, BertTokenizerFast

class FinBERTImpactRegressor(nn.Module):
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.regressor = nn.Linear(3, 1)

    def forward(self, input_ids, attention_mask):
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        return self.regressor(logits)          # linear output, no tanh

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = hf_hub_download("Kosmosas/BERT-bitcoin-sentiment",
                          "oof/finbert_oof_tone3_cond_fold6.pth")

model = FinBERTImpactRegressor()
model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
model.to(device).eval()
tokenizer = BertTokenizerFast.from_pretrained("yiyanghkust/finbert-tone")

# Fold 6 calibration, from its own validation window (see Calibration below).
VAL_MEAN, VAL_SD = -0.00883, 0.08335

texts = ["150 million dollars of long positions have been liquidated in the past 24 hours",
         "BlackRock files for a spot Bitcoin ETF"]
enc = tokenizer(texts, return_tensors="pt", truncation=True,
                padding="max_length", max_length=128).to(device)
with torch.no_grad():
    raw = model(enc["input_ids"], enc["attention_mask"]).squeeze(-1).cpu().numpy()

for t, r in zip(texts, raw):
    print(f"{(r - VAL_MEAN) / VAL_SD:+.3f}   {t}")
# -0.665   150 million dollars of long positions have been liquidated in the past 24 hours
# +0.491   BlackRock files for a spot Bitcoin ETF
```

Use `BertForSequenceClassification` explicitly. `AutoModelForSequenceClassification`
fails on this base checkpoint with recent `transformers` versions, because
`yiyanghkust/finbert-tone` ships a `config.json` without a `model_type` key.

## Calibration

Each fold is a separate model with its own output scale, so raw scores are not
comparable across folds. Every fold is standardised by the mean and standard
deviation of its own validation predictions — data that lies entirely before the
slice being scored, so this introduces no look-ahead.

| Fold | val mean | val sd |
|---|---|---|
| fold1 | −0.03661 | 0.13220 |
| fold2 | +0.05401 | 0.11166 |
| fold3 | +0.00785 | 0.09348 |
| fold4 | −0.07484 | 0.07614 |
| fold5 | −0.05916 | 0.08552 |
| **fold6** | **−0.00883** | **0.08335** |

## What the scores are worth

Out of sample the headline score carries a small but consistently signed
association with the short-horizon price response (r ≈ 0.076 on the final fold).
It does not support point forecasting of the next-hour price change: in the
accompanying paper no feature set containing it beats a zero-change forecast by a
margin that survives a Diebold–Mariano test. Treat the score as a weak
conditioning signal, not a predictor.

## License

Apache 2.0. The news and market data used for training carry their own terms.
