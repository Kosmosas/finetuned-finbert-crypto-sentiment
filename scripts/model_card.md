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

A fine-tuned **FinBERT** model for scoring the sentiment of financial / crypto
news headlines. A linear regression head with a `tanh` output is added on top of
the FinBERT base, producing a **continuous sentiment score in `[-1, 1]`**
(negative → positive) rather than discrete classes.

The model was trained to predict the short-horizon price impact of Bitcoin news
headlines, as part of the research released at
<https://github.com/Kosmosas> (see the accompanying paper).

## Files

- `fine-tuned-finbert.pth` — fine-tuned weights (PyTorch `state_dict`, ~419 MB).

## Usage

```python
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FinBERTRegression(nn.Module):
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.regressor = nn.Linear(3, 1)

    def forward(self, input_ids, attention_mask):
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits
        return torch.tanh(self.regressor(logits))

device = "cuda" if torch.cuda.is_available() else "cpu"
weights = hf_hub_download("Kosmosas/BERT-bitcoin-sentiment", "fine-tuned-finbert.pth")

model = FinBERTRegression()
model.load_state_dict(torch.load(weights, map_location=device))
model.to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

text = "The company's revenue exceeded expectations, leading to a positive outlook."
inputs = tokenizer(text, return_tensors="pt", truncation=True,
                   padding="max_length", max_length=256).to(device)
with torch.no_grad():
    score = model(inputs["input_ids"], inputs["attention_mask"]).item()
print(score)  # e.g. 0.87
```

> A CUDA-capable GPU is recommended; pass `map_location` to `torch.load` as shown
> when loading on a different device than the weights were saved on.

## License

Released under the Apache 2.0 license. Note that the underlying news/market data
used for training may carry its own usage terms.
