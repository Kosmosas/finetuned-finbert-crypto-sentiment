# Fine-tuned FinBERT — sentiment regressor

A minimal, self-contained example for scoring text with the fine-tuned FinBERT
model used in this work. **FinBERT** is the base model; it is extended with a
linear regression head and a `tanh` output, producing a continuous sentiment
score in **[-1, 1]** (negative → positive).

## Files

- `fine-tuned-finbert.pth` — fine-tuned weights (hosted externally, ~419 MB;
  see the repository [README](../README.md#data--model-files-hosted-outside-git)
  for download instructions).
- `sentiment_dataset_from_finbert.ipynb` — defines the model and scores text.

## Requirements

```bash
pip install torch transformers pandas tqdm
```

A CUDA-capable GPU is required.

## Usage

1. Download the weights so that `fine_tuned_BERT/fine-tuned-finbert.pth` exists
   (`python ../scripts/download_assets.py` from the repo root).
2. Open `sentiment_dataset_from_finbert.ipynb` and run the cells in order.
   The model is loaded with:

   ```python
   model = FinBERTRegression()
   model.load_state_dict(torch.load("fine-tuned-finbert.pth"))
   model.eval()
   tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
   ```
3. Edit the `text = "..."` line in the last cell and re-run it to score new text:

   ```
   Input text: The company's revenue exceeded expectations ...
   Sentiment score: 0.87
   ```
