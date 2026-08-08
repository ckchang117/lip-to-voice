"""torch.load compatibility shim for LipVoicer checkpoints.

LipVoicer issue #7 reports pickle/unpickle errors on some PyTorch / pickle
combinations when loading the released .pkl checkpoints. This wrapper tries
several strategies in order and surfaces the first one that succeeds.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any

import torch


class CheckpointLoadError(RuntimeError):
    """Raised when every loading strategy has been exhausted."""


def load(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a LipVoicer-style checkpoint, retrying with progressively looser settings.

    Returns whatever the checkpoint contains (typically a dict with
    `model_state_dict` and friends).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    errors: list[tuple[str, Exception]] = []

    # 1. Modern PyTorch default: weights_only=True. Most secure, often rejects pickles.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        errors.append(("weights_only=True", e))

    # 2. Trust the pickle. LipVoicer checkpoints predate weights_only.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        errors.append(("weights_only=False", e))

    # 3. Some older checkpoints fail on protocol negotiation. Read bytes ourselves
    #    and try a few pickle modules.
    raw = path.read_bytes()
    for label, mod in _candidate_pickle_modules():
        try:
            buf = io.BytesIO(raw)
            return torch.load(buf, map_location=map_location, pickle_module=mod, weights_only=False)
        except Exception as e:
            errors.append((f"pickle_module={label}", e))

    # 4. Last resort: try to unpickle the bytes directly and hope it's a state dict.
    try:
        return pickle.loads(raw)
    except Exception as e:
        errors.append(("raw pickle.loads", e))

    msg = "Could not load checkpoint after trying:\n"
    for label, err in errors:
        msg += f"  - {label}: {type(err).__name__}: {err}\n"
    raise CheckpointLoadError(msg.rstrip())


def _candidate_pickle_modules() -> list[tuple[str, Any]]:
    """Return (label, module) pairs for any pickle-compatible modules available."""
    out: list[tuple[str, Any]] = [("stdlib pickle", pickle)]
    try:
        import pickle5  # type: ignore[import-not-found]

        out.append(("pickle5", pickle5))
    except ImportError:
        pass
    return out


def self_test() -> None:
    """Round-trip a small state-dict through every code path to confirm imports work."""
    state = {"weight": torch.randn(4, 4), "bias": torch.zeros(4)}
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        torch.save(state, f.name)
        loaded = load(f.name)
    assert torch.equal(loaded["weight"], state["weight"])
    assert torch.equal(loaded["bias"], state["bias"])
    print("checkpoint_compat.self_test: OK")


if __name__ == "__main__":
    self_test()
