"""Restricted checkpoint loading with independently recorded metadata/hash."""
import json
import importlib
from pathlib import Path

import numpy as np
import torch

from .market import Market
from .neural import PriceNetwork
from .provenance import sha256


def load_model(metadata_path):
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = metadata_path.with_suffix(".pt")
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from recorded metadata")
    # Original new-run metadata contains NumPy float64 scalars, not arbitrary
    # Python classes. Restrict unpickling to torch's weights-only loader plus
    # these scalar types; never fall back to weights_only=False.
    numpy_core = "numpy.core.multiarray" if np.__version__.startswith("1.") else "numpy._core.multiarray"
    scalar = importlib.import_module(numpy_core).scalar
    dtype_type = type(np.dtype(np.float64))
    with torch.serialization.safe_globals([scalar, np.dtype, dtype_type]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embedded = json.loads(json.dumps(payload["metadata"]))
    for field in ("config", "market", "architecture", "seed"):
        if embedded[field] != metadata[field]:
            raise ValueError(f"Checkpoint metadata mismatch: {field}")
    market = Market(**metadata["market"])
    config = metadata["config"]
    model = PriceNetwork(market.d, config["width"], config["depth"], metadata["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, market, metadata
