from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ProtocolError


BUDGETS = (20, 50, 100, 200)
TARGET_PROPORTIONS = (0.30, 0.30, 0.25, 0.15)
CALIBRATION_RATIO = 0.30
CALIBRATION_SEED = 42


@dataclass(frozen=True)
class DynamicKThresholds:
    gallery_size: int
    tau1: float
    tau2: float
    tau3: float
    calibration_seed: int = CALIBRATION_SEED
    calibration_ratio: float = CALIBRATION_RATIO
    calibration_query_count: int = 0
    calibration_indices_sha256: str = ""
    source: str = "label-free coarse-margin quantiles"
    protocol: str = "u1652_fixed_threshold_dynamic_k_v1"

    def __post_init__(self) -> None:
        if not (self.tau1 < self.tau2 < self.tau3):
            raise ProtocolError("Thresholds must satisfy tau1 < tau2 < tau3")
        if self.gallery_size not in (100_000, 1_000_000):
            raise ProtocolError("Frozen paper thresholds are scale-specific for 100K or 1M")

    @property
    def ascending(self) -> tuple[float, float, float]:
        return self.tau1, self.tau2, self.tau3

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        payload["budgets"] = list(BUDGETS)
        payload["target_proportions"] = list(TARGET_PROPORTIONS)
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "DynamicKThresholds":
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "gallery_size",
            "tau1",
            "tau2",
            "tau3",
            "calibration_seed",
            "calibration_ratio",
            "calibration_query_count",
            "calibration_indices_sha256",
            "source",
            "protocol",
        }
        missing = required - payload.keys()
        if missing:
            raise ProtocolError(f"Threshold manifest is missing: {sorted(missing)}")
        return cls(**{key: payload[key] for key in required})


def calibration_indices(num_queries: int) -> tuple[np.ndarray, np.ndarray]:
    if num_queries <= 0:
        raise ProtocolError("num_queries must be positive")
    rng = np.random.default_rng(CALIBRATION_SEED)
    indices = np.arange(num_queries, dtype=np.int32)
    rng.shuffle(indices)
    count = int(round(num_queries * CALIBRATION_RATIO))
    return np.sort(indices[:count]), np.sort(indices[count:])


def calibrate_thresholds(calibration_margins: np.ndarray, gallery_size: int) -> DynamicKThresholds:
    margins = np.asarray(calibration_margins, dtype=np.float64).reshape(-1)
    if not len(margins) or not np.isfinite(margins).all():
        raise ProtocolError("Calibration margins must be finite and non-empty")
    tau1, tau2, tau3 = np.quantile(margins, [0.15, 0.40, 0.70], method="linear").tolist()
    return DynamicKThresholds(
        gallery_size=gallery_size,
        tau1=float(tau1),
        tau2=float(tau2),
        tau3=float(tau3),
        calibration_query_count=int(len(margins)),
    )


def assign_dynamic_k(margins: np.ndarray, thresholds: DynamicKThresholds) -> np.ndarray:
    values = np.asarray(margins, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ProtocolError("Dynamic-K margins must be finite")
    tau = np.asarray(thresholds.ascending, dtype=np.float64)
    levels = (values[:, None] < tau[None, :]).sum(axis=1)
    return np.asarray(BUDGETS, dtype=np.int32)[levels]


def distribution(assigned_k: np.ndarray) -> dict[str, Any]:
    assigned = np.asarray(assigned_k, dtype=np.int32).reshape(-1)
    if not np.isin(assigned, BUDGETS).all():
        raise ProtocolError(f"Assigned K must be one of {BUDGETS}")
    counts = {int(k): int(np.sum(assigned == k)) for k in BUDGETS}
    total = len(assigned)
    return {
        "query_count": total,
        "average_k": float(assigned.mean()),
        "counts": {str(k): counts[k] for k in BUDGETS},
        "ratios": {str(k): counts[k] / total for k in BUDGETS},
    }

