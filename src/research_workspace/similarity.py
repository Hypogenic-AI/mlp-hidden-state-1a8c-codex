"""Representation similarity utilities for hidden-state analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SVCCA_ROOT = PROJECT_ROOT / "code" / "svcca"
if str(SVCCA_ROOT) not in sys.path:
    sys.path.append(str(SVCCA_ROOT))

import cca_core  # type: ignore  # noqa: E402
import pwcca  # type: ignore  # noqa: E402


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA for matrices shaped [samples, features]."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xxt = x @ x.T
    yyt = y @ y.T
    hsic = np.sum(xxt * yyt)
    x_norm = np.linalg.norm(xxt)
    y_norm = np.linalg.norm(yyt)
    denom = x_norm * y_norm
    if denom == 0:
        return float("nan")
    return float(hsic / denom)


def samplewise_cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Mean cosine similarity across matched samples."""
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    denom = np.clip(x_norm * y_norm, 1e-12, None)
    cosine = np.sum(x * y, axis=1, keepdims=True) / denom
    return float(np.mean(cosine))


def svcca_similarity(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """SVCCA similarity with activations shaped [samples, features]."""
    x_rows = _center_rows(x.T)
    y_rows = _center_rows(y.T)
    result = cca_core.get_cca_similarity(
        x_rows,
        y_rows,
        epsilon=epsilon,
        compute_coefs=False,
        compute_dirns=False,
        verbose=False,
    )
    return float(np.mean(result["cca_coef1"]))


def pwcca_similarity(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-10) -> float:
    """PWCCA similarity with activations shaped [samples, features]."""
    x_rows = _center_rows(x.T)
    y_rows = _center_rows(y.T)
    pwcca_score, _, _ = pwcca.compute_pwcca(x_rows, y_rows, epsilon=epsilon)
    return float(pwcca_score)


def all_similarity_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute all configured similarity metrics for a pair of activation sets."""
    metrics = {
        "cka": linear_cka(x, y),
        "cosine": samplewise_cosine(x, y),
        "svcca": svcca_similarity(x, y),
        "pwcca": pwcca_similarity(x, y),
    }
    return metrics
