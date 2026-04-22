from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib


@dataclass(frozen=True)
class ModelSpec:
    model_path: Path
    scaler_path: Path | None = None


def load_pickle(path: Path) -> Any:
    """
    Load a model/scaler from disk.

    Preference:
    1) `joblib.load` (handles joblib-compressed files)
    2) `pickle.load` fallback
    """
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def infer_feature_names(model: Any, scaler: Any | None) -> list[str] | None:
    for obj in (model, scaler):
        if obj is None:
            continue
        names = getattr(obj, "feature_names_in_", None)
        if names is None:
            continue
        try:
            return [str(x) for x in list(names)]
        except Exception:
            pass
    return None


def infer_n_features(model: Any, scaler: Any | None) -> int | None:
    for obj in (model, scaler):
        if obj is None:
            continue
        n = getattr(obj, "n_features_in_", None)
        if isinstance(n, (int, np.integer)) and int(n) > 0:
            return int(n)
    return None


def get_classes(model: Any) -> list[str] | None:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    try:
        return [str(x) for x in list(classes)]
    except Exception:
        return None


def _to_frame(
    data: dict[str, float] | pd.DataFrame,
    feature_names: list[str] | None,
    n_features: int | None,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame([data])

    if feature_names is not None:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少必要欄位: {missing}")
        # Keep only needed columns, in a stable order.
        df = df[feature_names]
        return df

    if n_features is not None:
        # Accept either f0..f{n-1} or the first n numeric columns.
        expected = [f"f{i}" for i in range(n_features)]
        if all(c in df.columns for c in expected):
            return df[expected]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= n_features:
            return df[numeric_cols[:n_features]]

        raise ValueError(
            f"無法推斷特徵欄位。請提供欄位名 f0..f{n_features-1}，或使用含有至少 {n_features} 個數值欄位的 CSV。"
        )

    return df


def predict(
    model: Any,
    X: dict[str, float] | pd.DataFrame,
    *,
    scaler: Any | None = None,
    feature_names: list[str] | None = None,
    n_features: int | None = None,
) -> pd.DataFrame:
    df = _to_frame(X, feature_names=feature_names, n_features=n_features)

    X_arr: Any = df
    if scaler is not None:
        X_arr = scaler.transform(df)

    classes = get_classes(model)
    has_proba = hasattr(model, "predict_proba")

    if has_proba:
        proba = model.predict_proba(X_arr)
        proba = np.asarray(proba)
        if classes is None:
            classes = [str(i) for i in range(proba.shape[1])]
        pred_idx = np.argmax(proba, axis=1)
        pred_label = [classes[i] for i in pred_idx]

        out = pd.DataFrame({"pred": pred_label})
        for j, c in enumerate(classes):
            out[f"proba_{c}"] = proba[:, j]
        return pd.concat([df.reset_index(drop=True), out], axis=1)

    pred = model.predict(X_arr)
    pred = [str(x) for x in list(pred)]
    out = pd.DataFrame({"pred": pred})
    return pd.concat([df.reset_index(drop=True), out], axis=1)


def list_model_files(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted([p for p in models_dir.glob("*.joblib") if p.is_file()])


def describe_estimator(model: Any) -> str:
    t = type(model)
    return f"{t.__module__}.{t.__name__}"


def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")
