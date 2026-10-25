import json

import numpy as np
import pytest
import torch

from dgm_mc.checkpoint import load_model
from dgm_mc.market import equicorrelated
from dgm_mc.neural import PriceNetwork
from dgm_mc.provenance import sha256, write_json
from dgm_mc.report import load_verified, wilson


def test_reject_incomplete_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_verified(tmp_path)


def test_reject_changed_completed_artifact(tmp_path):
    (tmp_path / "records.json").write_text("[]")
    write_json(tmp_path / "completion.json", dict(status="complete", artifact_sha256={"records.json": "wrong"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified(tmp_path)


def test_safe_checkpoint_roundtrip_and_tamper(tmp_path):
    model = PriceNetwork(1, 8, 1)
    metadata = dict(config=dict(width=8, depth=1), architecture="dgm", seed=3,
                    market=equicorrelated(1).to_dict())
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(state_dict=model.state_dict(), metadata=metadata), checkpoint)
    metadata["checkpoint_sha256"] = sha256(checkpoint)
    write_json(tmp_path / "model.json", metadata)
    restored, market, _ = load_model(tmp_path / "model.json")
    assert market.d == 1
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    metadata["checkpoint_sha256"] = "bad"
    write_json(tmp_path / "model.json", metadata)
    with pytest.raises(ValueError, match="hash"):
        load_model(tmp_path / "model.json")


def test_wilson_endpoints_and_width():
    lo, hi = wilson(38, 40)
    assert lo < .95 < hi and hi - lo > .1
    lo, hi = wilson(0, 40)
    assert lo == pytest.approx(0) and 0 < hi < .1
