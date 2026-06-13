import torch
import numpy as np
from iTransformer import iTransformer
from typing import Tuple, Optional

def load_model(model_path: str,
               num_variates: int,
               lookback_len: int = 24*7,
               dim: int = 128,
               depth: int = 2,
               heads: int = 32,
               dim_head: int = 32,
               pred_length: int = 1,
               num_tokens_per_variate: int = 1,
               use_reversible_instance_norm: bool = True,
               device: str = None) -> iTransformer:
    """Load a trained iTransformer model"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = iTransformer(
        num_variates=num_variates,
        lookback_len=lookback_len,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        pred_length=pred_length,
        num_tokens_per_variate=num_tokens_per_variate,
        use_reversible_instance_norm=use_reversible_instance_norm
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
    model.eval()

    return model, device

def make_predictions(model: iTransformer,
                    X_tensor: torch.Tensor,
                    device: str,
                    target_col_index: int) -> np.ndarray:
    """Make predictions with the model"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy().squeeze()

    # Extract the target column predictions
    # For iTransformer, predictions shape is (batch_size, pred_length, num_variates)
    # We assume the target is the first variate (index 0)
    if len(predictions.shape) > 1:
        pred_changes = predictions[:, 0]  # Assume price change is the first variate
    else:
        pred_changes = predictions

    return pred_changes

def prepare_tensor(X_val: np.ndarray, device: str) -> torch.Tensor:
    """Convert numpy array to torch tensor on device"""
    return torch.tensor(X_val, dtype=torch.float32).to(device)